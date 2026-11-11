"""Interactive raised-wheel mapping using the independent bounded serial probe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/home/pi/docker/tmp/mentorpi_nav')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel',type=int,choices=range(1,5),help='Test only this channel; default tests all four')
    args = parser.parse_args()
    channels = [args.channel] if args.channel else [1,2,3,4]
    notes_file = ROOT/'diagnostics/wheel_channel_notes.json'
    notes = json.loads(notes_file.read_text()) if notes_file.exists() else {'channels':{}}
    print('Each pulse: positive 0.6 RPS for 0.5 seconds, then automatic stop.')
    print('Keep the chassis securely supported with ALL wheels clear. Keep hands away from wheels.')
    print('Navigation, factory app and Wi-Fi helper must remain stopped; the probe refuses competing serial owners.')
    if input('Type RAISED to confirm, or anything else to quit: ').strip() != 'RAISED':
        print('Cancelled; no motor commands sent.')
        return
    for channel in channels:
        while True:
            answer = input(f'Channel {channel}: Enter=pulse, s=skip, q=quit: ').strip().lower()
            if answer == 'q':
                return
            if answer == 's':
                break
            if answer:
                continue
            print(f'Pulsing channel {channel} now...',flush=True)
            run = subprocess.run([sys.executable,str(ROOT/'raw_rrc_probe.py'),
                '--motor',str(channel),'--rps','0.6','--duration','0.5','--wheels-raised'],
                capture_output=True,text=True)
            if run.returncode or 'PARENT_STOP: explicit zero packets sent' not in run.stdout:
                print(run.stdout)
                print(run.stderr)
                raise RuntimeError('Probe did not complete successfully; inspect output before continuing')
            print('Pulse complete; explicit stop packets sent.')
            observation = input('Note wheel + direction (example: front right forward), or r=repeat: ').strip()
            if observation.lower() == 'r':
                continue
            if not observation:
                print('No note saved; repeat or skip this channel.')
                continue
            notes['channels'][str(channel)] = {'positive_rps':.6,'duration_s':.5,
                'observation':observation,'recorded_utc':datetime.now(timezone.utc).isoformat()}
            notes_file.parent.mkdir(parents=True,exist_ok=True)
            notes_file.write_text(json.dumps(notes,indent=2))
            print('Saved.')
            break
    print('\nCopy these results back into chat:')
    for channel,entry in sorted(notes['channels'].items()):
        print(f'Channel {channel}: {entry["observation"]} (positive command)')
    print('Notes file:',notes_file)

if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nTester closed. Any active probe has its own deadline and stop teardown.')
        raise SystemExit(130)
