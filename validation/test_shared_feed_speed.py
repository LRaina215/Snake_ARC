"""Check the user speed knob reaches manual and automatic material advancement; no COM."""
import contextlib
import io
import unittest
from unittest.mock import patch

from validation.test_real_controller import real, prepared_controller
from snake_control import gravity_handoff as library


class SharedFeedSpeedTests(unittest.TestCase):
    def check_routes(self, rate, overrides, effective):
        with patch.object(real,'FEED_RATE_JOINTS_S',rate), patch.object(real,'FEED_OPTIONS_OVERRIDE',overrides), \
                patch.object(real,'HELIX_TRANSITION_SPEED_SCALE',1.3), contextlib.redirect_stdout(io.StringIO()):
            manual,_=prepared_controller()
            manual.command('gait_feeding');manual.command('step')
            self.assertAlmostEqual(manual.feeding.options.maximum_position_rate_joints_s,effective)

            automatic,_=prepared_controller()
            embedded=automatic.read_gravity_entry(False,helix_only=True)
            offline=library.GravityCrossing(real.LINK_LENGTHS_M,automatic.enabled,automatic.measured_angles(),
                options=library.GravityCrossingOptions(feed_rate_joints_s=effective,transition_speed_scale=1.3),
                helix_only=True)
            for module_name,core in [('standalone',embedded),('library',offline)]:
                with self.subTest(module=module_name):
                    front=core.front
                    self.assertIsNotNone(front.transfer)
                    helper=front.transfer
                    before=helper.position
                    # Execute the automatic adjustment, including its forced-step
                    # path, so a leftover hardcoded .1 cannot pass this check.
                    front.feedback_time=0.
                    front._advance_transfer(.02)
                    self.assertAlmostEqual(abs(helper.position-before),effective*1.3*.02)
                    self.assertAlmostEqual(core.status()['auto_feed_rate_joints_s'],effective*1.3)

    def test_top_level_speed_reaches_manual_and_auto_with_default_timing_unchanged(self):
        for rate in (.1,.2):
            with self.subTest(rate=rate):
                self.check_routes(rate,{},rate)

    def test_previously_documented_override_is_also_shared(self):
        self.check_routes(.2,{'maximum_position_rate_joints_s':.15},.15)


if __name__=='__main__':unittest.main()
