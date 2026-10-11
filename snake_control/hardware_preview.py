"""Offline mapping to the existing EXE profile and Feetech sync-write format.

This is a payload preview, not a live serial owner or verified hardware driver.
Reference: Adaptive_Pole_Climbing_Real.py and FeetechProtocol.cs.
"""
import json
from pathlib import Path
import numpy as np


def sync_write(address, width, values):
    if not values or not 0 <= address <= 255 or not 1 <= width <= 255:
        raise ValueError('Invalid sync-write layout')
    body = bytearray([0xFE, 0, 0x83, address, width])
    for servo_id, value in values:
        if not 1 <= servo_id <= 253 or len(value) != width:
            raise ValueError('Invalid servo ID or register width')
        body.extend([servo_id, *value])
    if len(body) - 1 > 255:
        raise ValueError('Packet exceeds protocol length')
    body[1] = len(body) - 1
    return b'\xff\xff' + bytes(body) + bytes([(~sum(body)) & 255])


def preview_frame(parameters, phase, profile_path):
    profile = json.loads(Path(profile_path).read_text(encoding='utf-8-sig'))
    joints = sorted(profile['Joints'], key=lambda j: j['Index'])
    if [j['Index'] for j in joints] != list(range(1, parameters.joints + 1)):
        raise ValueError('Hardware profile joint count/order mismatch')
    ids = [j['ServoId'] for j in joints]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate servo IDs')
    controller = parameters.targets(phase)
    degrees = np.degrees(controller) * np.where(np.arange(parameters.joints) % 2, -1, 1)
    positions, active_targets = [], []
    for j, deg, enabled in zip(joints, degrees, parameters.enabled):
        if not j['MinimumAngle'] <= deg <= j['MaximumAngle'] or j['Direction'] not in (-1, 1):
            raise ValueError('Invalid hardware angle or direction')
        raw = round(j['Zero'] + j['Direction'] * deg * 4096 / 360)
        if not 0 <= raw <= 4095:
            raise ValueError('Target outside single-turn position range')
        positions.append(raw)
        if enabled:
            # Position, duration=0, running speed=800; mirrors existing EXE layout.
            active_targets.append((j['ServoId'], raw.to_bytes(2, 'little') + b'\x00\x00\x20\x03'))
    mask = sync_write(0x28, 1, [(j['ServoId'], bytes([int(e)])) for j, e in zip(joints, parameters.enabled)])
    return dict(phase_rad=float(phase), enabled=parameters.enabled.tolist(),
                target_controller_rad=controller.tolist(), target_protocol_deg=degrees.tolist(),
                servo_ids=ids, positions_raw=positions, torque_mask_packet_hex=mask.hex(' '),
                active_target_packet_hex=sync_write(0x2A, 6, active_targets).hex(' '),
                reference_port=profile['Port'], reference_baud=profile['BaudRate'],
                execution='offline_preview_only')
