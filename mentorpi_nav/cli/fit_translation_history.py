"""Fit signed translation memory on separate runs, using recorded applied commands."""
import argparse
import copy
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',required=True);parser.add_argument('--static-profile',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    import numpy as np
    from scipy.optimize import differential_evolution
    from mentorpi_nav.translation_response import AxisTranslationResponse,DIRECTIONS
    prepared=[]
    for path in sorted(Path(args.captures).glob('run_*/block_*.json')):
        record=json.loads(path.read_text())
        if not record['passed']:raise ValueError('Failed recording')
        trace=record['trace'];times=np.array([r['time'] for r in trace]);dt=np.diff(times)
        commands=np.array([r['guard']['applied'][:2] for r in trace[:-1]])
        windows=[]
        for i,pulse in enumerate(record['pulses']):
            end=record['pulses'][i+1]['start_time'] if i+1<len(record['pulses']) else times[-1]
            axis=int(pulse['direction'] in ('left','right'));sign=1 if pulse['direction'] in ('forward','left') else -1
            windows.append((axis,sign,np.flatnonzero((times[:-1]>=pulse['start_time'])&(times[:-1]<end)),sign*pulse['laser_body_delta'][axis],pulse['direction']))
        prepared.append((path.parent.name,dt,commands,windows,str(path)))
    base=json.loads(Path(args.static_profile).read_text());profile=copy.deepcopy(base['profile'])
    profile.update(version='translation_history_v1',breakaway_distance_m=[.01,.01],idle_memory_decay_s=10.)
    def apply(parameters,axis):
        p=copy.deepcopy(profile)
        for offset in (0,1):
            p['directions'][DIRECTIONS[2*axis+offset]]={'gain':float(parameters[2*offset]),'deadband_mps':float(parameters[2*offset+1])}
        p['breakaway_distance_m'][axis]=float(parameters[4]);return p
    def predict(row,p,axis):
        _,dt,commands,windows,_=row;model=AxisTranslationResponse(p,axis)
        distances=np.array([model.step(float(u),float(t))*t for u,t in zip(commands[:,axis],dt)])
        return [(sign*float(distances[index].sum()),measured,name) for a,sign,index,measured,name in windows if a==axis]
    for axis in (0,1):
        fitting=[r for r in prepared if r[0] in ('run_01','run_02')]
        def objective(parameters):
            p=apply(parameters,axis)
            errors=[(prediction-measured)**2 for row in fitting for prediction,measured,_ in predict(row,p,axis)]
            return sum(errors)/len(errors)
        solution=differential_evolution(objective,[(.6,1.6),(0,.035),(.6,1.6),(0,.035),(.001,.025)],seed=73003+axis,popsize=7,maxiter=45,tol=.003,polish=False)
        profile=apply(solution.x,axis)
    results=[]
    for row in prepared:
        for axis in (0,1):
            for prediction,measured,name in predict(row,profile,axis):
                results.append({'run':row[0],'direction':name,'predicted_m':prediction,'measured_m':measured,'error_m':prediction-measured})
    held=[r for r in results if r['run']=='run_03']
    report=base|{'profile':profile,'results':results,'validation':{'n':len(held),'rmse_m':float(np.sqrt(np.mean([r['error_m']**2 for r in held]))),'max_abs_error_m':max(abs(r['error_m']) for r in held)}}
    report['limits']+=['Signed memory empirical; 10-second idle decay and cold block start are assumed, not identified.','No long-duration or diagonal hardware validation.']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'translation_profile.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'profile':profile,'validation':report['validation']},indent=2))


if __name__=='__main__':main()
