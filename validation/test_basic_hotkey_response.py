"""Actual pause-state replay and physical-key identity checks; no serial port or OS hook."""
import contextlib
import io
import json
from pathlib import Path
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pynput.keyboard import Key, KeyCode

from validation.test_real_controller import real, RecordingSerial
from codes import Rolling_Arc_Debug_Mujoco as debug
from snake_control.rolling_arc import ArcParameters

DATA=json.loads((Path(__file__).parent/'fixtures/basic_key_pause_20261108.json').read_text(encoding='utf8'))


def recorded_controller(module=real,sim=False):
    row=DATA['before_reformation'];s=row['basic_arc_reference']
    bus=module.FeetechBus(module.DryRunSerial())
    if sim:
        session=SimpleNamespace(sim=SimpleNamespace(p=ArcParameters(joints=18)),elapsed=0.)
        c=debug.DebugArcController(bus,session)
    else:c=module.RollingArcController(bus)
    c.targets=row['target_controller_deg'].copy();c.enabled=row['enabled'].copy()
    c.phase=c.command_phase=row['phase_rad'];c.pending_phase=row['pending_phase_rad']
    c.mode=row['mode'];c.first,c.last=row['window']
    c.amplitude,c.omega,c.phase_step=row['amplitude_deg'],row['omega_rad_s'],row['phase_step_rad']
    c.basic_arc_reference=module.ArcCurveReference(module.LINK_LENGTHS_M,s['head_m'],s['arc_length_m'],
        s['radius_m'],s['reference_amplitude_deg'])
    return c


class BasicHotkeyResponseTests(unittest.TestCase):
    def test_recorded_repeated_keys_advance_phase_without_reformation_or_settle(self):
        self.assertGreater(DATA['after_reformation']['settle_remaining_s'],0.)
        for module,sim in [(real,False),(debug.real,True)]:
            with self.subTest(module=module.__name__),contextlib.redirect_stdout(io.StringIO()):
                c=recorded_controller(module,sim)
                self.assertGreater(np.max(abs(np.array(c.basic_targets())-c.targets)),1e-6)
                for command in ['back','back','step','back','step','step']:
                    before=c.phase;c.command(command)
                    self.assertFalse(c.forming);self.assertEqual(c.settle_remaining,0.)
                    expected=float(np.clip(c.pending_phase,-c.omega*.02,c.omega*.02));c.tick(.02)
                    self.assertAlmostEqual(c.phase-before,expected)
                    self.assertNotEqual(c.phase,before)

    def test_existing_step_size_configuration_and_hotkeys_apply_in_both_entries(self):
        for module,sim in [(real,False),(debug.real,True)]:
            with self.subTest(module=module.__name__):
                with patch.object(module,'PHASE_STEP_RAD',.3):
                    self.assertEqual(module.RollingArcController(module.FeetechBus(module.DryRunSerial())).phase_step,.3)
                c=recorded_controller(module,sim);c.mode='hold';c.pending_phase=0.
                self.assertEqual(module.decode_hotkey(187,True),'delta+')
                self.assertEqual(module.decode_hotkey(189,True),'delta-')
                c.command('delta+');self.assertAlmostEqual(c.phase_step,.2)
                c.command('step');self.assertAlmostEqual(c.pending_phase,.2)
                c.command('pause');c.command('delta-');c.command('back')
                self.assertAlmostEqual(c.pending_phase,-.1)

    def test_step_finishes_on_phase_budget_and_target_can_settle_without_blocking_next_key(self):
        c=recorded_controller();c.mode='hold';c.pending_phase=0.;c.freeze_targets=True
        with patch.object(real,'MAX_TARGET_RATE_DEG_S',1.):
            c.command('step');start=c.phase
            for _ in range(9):c.tick(.02)
            self.assertEqual(c.mode,'hold');self.assertAlmostEqual(c.phase-start,c.phase_step)
            self.assertGreater(np.max(abs(np.array(c.basic_targets())-c.targets)),.1)
            previous=c.targets.copy();c.tick(.02)
            self.assertNotEqual(c.targets,previous)  # A fixed phase target keeps approaching, without gating H.
            phase=c.phase;c.command('back');c.tick(.02)
            self.assertLess(c.phase,phase);self.assertFalse(c.forming);self.assertEqual(c.settle_remaining,0.)
            c.command('pause');held=c.targets.copy();c.tick(.02)
            self.assertEqual(c.targets,held)  # Explicit pause freezes immediately.

    def test_ctrl_release_before_letter_does_not_leave_j_or_h_stuck_in_pressed_cache(self):
        listeners=[]
        class Listener:
            def __init__(self,**kwargs):self.callbacks=kwargs;listeners.append(self)
            def start(self):pass
        keyboard=SimpleNamespace(Key=Key,Listener=Listener);commands=queue.Queue()
        real.global_key_listener(commands,keyboard)
        press=listeners[0].callbacks['on_press'];release=listeners[0].callbacks['on_release']
        for vk,control_char,letter in [(74,'\n','j'),(72,'\b','h')]:
            for _ in range(2):
                press(Key.ctrl_l);press(KeyCode(vk=vk,char=control_char))
                press(KeyCode(vk=vk,char=control_char))  # Held-key OS repeat stays one action.
                release(Key.ctrl_l);release(KeyCode(vk=vk,char=letter))
        got=[]
        while not commands.empty():got.append(commands.get_nowait())
        self.assertEqual(got,['step','step','back','back'])


if __name__=='__main__':unittest.main()
