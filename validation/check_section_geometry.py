"""One batch of eight meaningful migration fixtures; optional C# oracle build."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess,argparse
import numpy as np
from snake_control.section_geometry import forward,paper_axis,projection_basis,continuous_section
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build-oracle',action='store_true')
    args=parser.parse_args();out=ROOT/'validation/reports/fixed_section/r0_geometry';out.mkdir(parents=True,exist_ok=True)
    p=ArcParameters();fixtures=[]
    for first in (8,7):
        for phase in (0,np.pi/4,np.pi/2,3*np.pi/4):
            q=np.where((np.arange(20)>=first-1)&(np.arange(20)<=first+4),
                       np.radians(60)*np.where(np.arange(20)%2==0,np.cos(phase),np.sin(phase)),0)
            fixtures.append(dict(name=f'{first}_{phase:.2f}',q=q.tolist(),lengths=[.0645]*21,first=first,last=first+5))
    fixture_path=out/'fixtures.json';fixture_path.write_text(json.dumps(fixtures),encoding='utf-8')
    if args.build_oracle:
        ref=ROOT.parent/'Snake_Robot_Gait/visualizer/windows_app';exe=ROOT/'tmp/GeometryOracle.exe';exe.parent.mkdir(exist_ok=True)
        sources=[ref/name for name in ('Models.cs','Kinematics.cs','RollingAxis.cs','CrossSectionCurve.cs','FeetechProtocol.cs')]
        subprocess.run([r'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe','/nologo','/target:exe',
                        '/codepage:65001','/out:'+str(exe),'/reference:System.dll','/reference:System.Core.dll',
                        '/reference:System.Drawing.dll','/reference:System.Web.Extensions.dll',str(ROOT/'validation/reference_geometry.cs'),
                        *map(str,sources)],check=True)
        subprocess.run([str(exe),str(fixture_path),str(out/'oracle.json')],check=True)
    oracle=json.loads((out/'oracle.json').read_text(encoding='utf-8-sig'));rows=[]
    for f,o in zip(fixtures,oracle):
        points=forward(f['q'],f['lengths']);local=points[f['first']-1:f['last']+3]
        axis=paper_axis(local);xy=(local-axis['point'])@projection_basis(axis['direction'])[:,:2]
        curve=continuous_section(xy,original=True)
        row=dict(name=f['name'],fk_error_m=float(np.max(abs(points-np.array(o['points'])/1000))),
                 axis_difference_deg=float(np.degrees(np.arccos(np.clip(abs(axis['direction']@o['axis']),0,1)))),
                 variance_error=abs(axis['variance']-o['variance']),segment_count_match=len(curve)==len(o['segments']))
        for key,scale in [('length',1000),('radius',1000),('turn',1)]:
            row[key+'_maximum_error']=float(np.max(abs(np.array([getattr(s,key) for s in curve])-np.array([s[key] for s in o['segments']])/scale)))
        row['passed']=row['segment_count_match'] and row['fk_error_m']<1e-9 and row['axis_difference_deg']<.01 and row['turn_maximum_error']<1e-8 and row['length_maximum_error']<1e-9 and row['radius_maximum_error']<1e-8
        rows.append(row)
    (out/'comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('Migration fixtures:',len(rows),'passed:',sum(r['passed'] for r in rows))
    if not all(r['passed'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
