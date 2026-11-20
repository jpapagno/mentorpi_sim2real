"""Fit settled translation distances and validate on separate capture runs."""
import argparse
import hashlib
import json
from pathlib import Path
import math


def samples(root):
    rows=[]
    for filename in sorted(Path(root).glob('run_*/block_*.json')):
        record=json.loads(filename.read_text())
        if not record.get('passed'):raise ValueError(f'Aborted capture: {filename}')
        pulses=record['pulses']
        for index,pulse in enumerate(pulses):
            name=pulse['direction']
            if name not in ('forward','backward','left','right'):raise ValueError('Translation recordings only')
            axis=int(name in ('left','right'));sign=1 if name in ('forward','left') else -1
            end=pulses[index+1]['start_time'] if index+1<len(pulses) else record['trace'][-1]['time']
            trace=[r for r in record['trace'] if pulse['start_time']<=r['time']<=end]
            intervals=[(max(0.,sign*a['guard']['applied'][axis]),b['time']-a['time']) for a,b in zip(trace,trace[1:])]
            rows.append({'run':filename.parent.name,'file':str(filename),'slot':index,'direction':name,'axis':axis,
                         'measured_m':sign*pulse['laser_body_delta'][axis],
                         'cross_axis_m':pulse['laser_body_delta'][1-axis],
                         'gyro_angle_rad':pulse['gyro_angle_radians'],'intervals':intervals})
    return rows


def fit(rows,fit_runs):
    import numpy as np
    from scipy.optimize import least_squares
    from mentorpi_nav.translation_response import DIRECTIONS,validate_translation_response
    profile={'version':'translation_deadzone_v1','directions':{}}
    for name in DIRECTIONS:
        selected=[r for r in rows if r['run'] in fit_runs and r['direction']==name]
        if len(selected)<6:raise ValueError('Need repeats at three speeds per direction')
        def errors(p):
            return [sum(p[0]*max(0,u-p[1])*dt for u,dt in r['intervals'])-r['measured_m'] for r in selected]
        solution=least_squares(errors,[1.,.01],bounds=([.1,0],[3,.08]))
        profile['directions'][name]={'gain':float(solution.x[0]),'deadband_mps':float(solution.x[1])}
    validate_translation_response(profile)
    results=[]
    for row in rows:
        params=profile['directions'][row['direction']]
        predicted=sum(params['gain']*max(0,u-params['deadband_mps'])*dt for u,dt in row['intervals'])
        ideal=sum(u*dt for u,dt in row['intervals'])
        results.append({k:v for k,v in row.items() if k!='intervals'}|{'split':'fit' if row['run'] in fit_runs else 'validation',
                       'predicted_m':predicted,'ideal_m':ideal,'error_m':predicted-row['measured_m']})
    held=[r for r in results if r['split']=='validation']
    if not held:raise ValueError('Separate held-out run required')
    return profile,results,{'n':len(held),'rmse_m':float(np.sqrt(np.mean([r['error_m']**2 for r in held]))),
                            'max_abs_error_m':max(abs(r['error_m']) for r in held),
                            'ideal_rmse_m':float(np.sqrt(np.mean([(r['ideal_m']-r['measured_m'])**2 for r in held]))),
                            'max_abs_cross_axis_m':max(abs(r['cross_axis_m']) for r in held),
                            'max_abs_gyro_deg':max(abs(math.degrees(r['gyro_angle_rad'])) for r in held)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',required=True);parser.add_argument('--fit-runs',nargs='+',default=['run_01','run_02'])
    parser.add_argument('--output',required=True);args=parser.parse_args()
    rows=samples(args.captures);profile,results,validation=fit(rows,args.fit_runs)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report={'profile':profile,'fit_runs':args.fit_runs,'validation':validation,'results':results,
            'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(args.captures).glob('run_*/block_*.json')},
            'scope':'one-second pure-axis translation, .06/.08/.1m/s, one surface; distances from RF2O, no independent metric-scale check',
            'limits':['settled endpoint fit, not startup/stop timing identification','no diagonal or combined-axis validation',
                      'no voltage/surface sweep; no buzzing/PWM identification','fit applied velocity, not requested velocity; shared command ramps retained']}
    (out/'translation_profile.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'profile':profile,'validation':validation},indent=2))


if __name__=='__main__':main()
