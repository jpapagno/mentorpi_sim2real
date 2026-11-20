"""Fit body-response profiles and validate against separate recorded pulses."""
import argparse
import hashlib
import json
from pathlib import Path
from mentorpi_nav.robot.motion_profile import fit_profile,pulse_samples,predicted_rate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',nargs='+',required=True)
    p.add_argument('--validate',nargs='+',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    fit_paths=[Path(v).resolve() for v in args.fit];validation_paths=[Path(v).resolve() for v in args.validate]
    if set(fit_paths)&set(validation_paths):p.error('Fitting and validation recordings must be separate')
    def read(paths):
        samples=[]
        for path in paths:
            for sample in pulse_samples(json.loads(path.read_text())):
                samples.append({**sample,'recording':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return samples
    fitting=read(fit_paths);heldout=read(validation_paths);profile=fit_profile(fitting)
    rows=[]
    for sample in heldout:
        prediction=predicted_rate(profile['directions'][sample['direction']]['knots'],sample['command'])
        rows.append({**sample,'predicted_response':prediction,'prediction_error':prediction-sample['response']})
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'profile.json').write_text(json.dumps(profile,indent=2))
    (output/'fitting_samples.json').write_text(json.dumps(fitting,indent=2))
    (output/'validation.json').write_text(json.dumps({'kind':'held-out pulse recordings; not closed-loop target validation','samples':rows},indent=2))
    print(json.dumps({'profile':str(output/'profile.json'),'fit_samples':len(fitting),'validation_samples':len(rows),'hardware_target_validation':False}))


if __name__=='__main__':main()
