"""Compare observers on complete saved real runs, without executing commands.

The recorded masks and angles are fixed inputs, not a new closed-loop trial.
Fresh original rows are archived for reproducible evidence; no hardware IO.
"""
import gzip
import json
from pathlib import Path
import subprocess
import sys
import types
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from snake_control.whole_feeding_observer import WholeFeedingObserver
from snake_control.continuous_feeding import ContinuousFeeding,ContinuousFeedingOptions

REPORT=ROOT/'validation/reports/feeding_reference_recovery_20261021'


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    old=types.ModuleType('snake_control._prior_feeding_observer')
    old.__package__='snake_control';sys.modules[old.__name__]=old
    exec(subprocess.check_output(['git','show','efa9e17:snake_control/whole_feeding_observer.py']).decode('utf8'),old.__dict__)
    summary=[]
    for filename in ('rolling_arc_20261021_132633_104913.jsonl','rolling_arc_20261021_133912_480257.jsonl'):
        archive=REPORT/(Path(filename).stem+'.fresh.jsonl.gz')
        source=ROOT/'codes/logs'/filename
        rows=[]
        if not archive.exists() and source.exists():
            last=None
            with source.open(encoding='utf8') as stream,gzip.open(archive,'wt',encoding='utf8') as output:
                for number,line in enumerate(stream,1):
                    r=json.loads(line);m=r.get('feeding_motor_input');f=r.get('feeding')
                    if not m or not f:continue
                    key=(m['feedback_time_s'],m['control_time_s'])
                    if last is not None and key[0]==last[0]:continue
                    last=key
                    record=dict(line=number,row=r)
                    output.write(json.dumps(record,ensure_ascii=False)+'\n');rows.append(record)
        else:
            with gzip.open(archive,'rt',encoding='utf8') as stream:rows=list(map(json.loads,stream))
        last=None;observers=None;counters=[Counter(),Counter()];events=Counter();examples=[]
        elapsed_invalid=[0.,0.];longest=[0.,0.];intervals=[];acquisitions={}
        gaps=0;maximum_offset=maximum_correction=0.
        for record in rows:
            r=record['row'];f=r['feeding'];m=r['feeding_motor_input'];t=m['feedback_time_s']
            reset=last is None or t<last
            if reset:
                window=(f['options']['first_joint'],f['options']['last_joint'])
                observers=[old.WholeFeedingObserver(np.full(21,.0645),window),WholeFeedingObserver(np.full(21,.0645),window)]
                elapsed_invalid=[0.,0.]
            elif t>last:
                intervals.append(t-last)
                if t-last>.6:
                    gaps+=1;elapsed_invalid=[0.,0.]
            observations=[o.observe(m['angles_deg'],tuple(f['window']),t) for o in observers]
            for i,o in enumerate(observations):
                counters[i]['samples']+=1;counters[i]['valid' if o['valid'] else o.get('reason','invalid')]+=1
                elapsed_invalid[i]=0. if o['valid'] else elapsed_invalid[i]+(0. if reset or t-last>.6 else t-last)
                longest[i]=max(longest[i],elapsed_invalid[i])
            current=observations[1];quality=current.get('end_quality',{})
            if current['valid']:
                events[quality.get('reference_event')]+=1
                maximum_offset=max(maximum_offset,abs(quality['coordinate_offset_m']))
                maximum_correction=max(maximum_correction,abs(quality['handover_correction_m']))
            if observations[0]['valid']!=current['valid'] and len(examples)<8:
                examples.append(dict(line=record['line'],time_s=r['time_s'],
                    old=observations[0],new=current))
            task=str(r['feeding_action_id']);entry=acquisitions.get(task)
            if f['stage']=='acquire' or (f['stage']=='feed_out' and entry is not None and not entry.get('departed',False)):
                if entry is None:
                    entry=dict(core=ContinuousFeeding(np.full(21,.0645),ContinuousFeedingOptions(**f['options'])),samples=[],
                        rows=0,blocked_tracking=0,untrusted=0,first_ready_line=None,max_error_deg=0.,worst_joint=None,
                        last_metrics=None,reasons=Counter())
                    acquisitions[task]=entry
                if f['stage']=='feed_out':entry['departed']=True
                q=np.array(m['angles_deg']);mask=np.array(r['enabled'],bool)
                desired=entry['core'].profile(f['position_joints'],f['bearing_bias'])[0]
                error=np.where(mask,abs(np.degrees(desired)-q),-1.)
                j=int(error.argmax());entry['rows']+=1
                if error[j]>entry['max_error_deg']:
                    entry.update(max_error_deg=float(error[j]),worst_joint=j+1,
                        target_deg=float(np.degrees(desired[j])),actual_deg=float(q[j]))
                if error[j]>entry['core'].options.tracking_tolerance_deg:
                    entry['blocked_tracking']+=1;entry['samples'].clear()
                elif not current['valid']:
                    entry['untrusted']+=1;entry['samples'].clear()
                else:
                    metrics,reason=entry['core'].collect_length_datum(entry['samples'],q,current['transfer_coordinate_m'],quality['reference_epoch'],t)
                    entry['last_metrics']=metrics;entry['reasons'][reason or 'ready']+=1
                    if reason is None and entry['first_ready_line'] is None:entry['first_ready_line']=record['line']
            last=t
        public={key:{k:v for k,v in value.items() if k not in ('core','samples')} for key,value in acquisitions.items()}
        summary.append(dict(source=filename,archived_fresh_rows=len(rows),old=dict(counters[0]),new=dict(counters[1]),
            longest_invalid_sample_span_excluding_gaps_s=longest,feedback_gaps_over_point6s=gaps,
            maximum_coordinate_offset_mm=1000*maximum_offset,maximum_handover_correction_mm=1000*maximum_correction,
            reference_events=dict(events),changed_examples=examples,
            feedback_interval_percentiles_s=np.percentile(intervals,[0,50,95,100]).tolist(),acquisition_on_recorded_pose=public))
    (REPORT/'replay.json').write_text(json.dumps(dict(baseline='efa9e17',scope='fixed recorded angles/masks; no new closed-loop physics or real actions',runs=summary),ensure_ascii=False,indent=2),encoding='utf8')
    for result in summary:
        print(result['source'],result['old'],result['new'],result['reference_events'],flush=True)


if __name__=='__main__':main()
