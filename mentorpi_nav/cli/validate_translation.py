"""Replay recorded requests through shared simulator; no hardware access."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',required=True);parser.add_argument('--scenario-file',required=True)
    parser.add_argument('--run',default='run_03');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from mentorpi_nav.core import NavigationCfg,NavigationState
    torch.set_num_threads(1)
    cfg=NavigationCfg(**json.loads(Path(args.scenario_file).read_text()))
    cfg=replace(cfg,scene_spec=None,obstacle_count=0,arena_half_extent=20,actuator_scale_range=None)
    results=[]
    for filename in sorted((Path(args.captures)/args.run).glob('block_*.json')):
        record=json.loads(filename.read_text())
        if not record['passed']:raise ValueError('Cannot validate aborted capture')
        trace=record['trace'];timestamps=np.array([r['time'] for r in trace]);start=timestamps[0]
        times=np.arange(start,timestamps[-1]+cfg.dt,cfg.dt)
        requests=np.array([r['requested'] for r in trace]);indices=np.searchsorted(timestamps,times,side='right').clip(1,len(trace))-1
        trajectories={}
        for label,settings in [('calibrated',cfg),('baseline',replace(cfg,translation_response=None))]:
            state=NavigationState(1,settings);state.goal[:]=10
            positions=[]
            for index in indices:
                positions.append(state.xy[0].tolist())
                # The diagnostic runner disarms during holds. Hardware stop
                # bypasses command/wheel slew; reproduce that explicit event.
                # Normal enabled policy zero-actions still retain their ramps.
                if not trace[index]['enabled']:
                    state.body_command_velocity.zero_();state.wheel_rps.zero_()
                    state.velocity.zero_();state.delayed_actions.zero_()
                action=torch.tensor(requests[index]/np.array(cfg.speed_limits),dtype=torch.float32)[None]
                state.step(action)
            trajectories[label]=np.array(positions)
        for i,pulse in enumerate(record['pulses']):
            a=pulse['start_time'];b=record['pulses'][i+1]['start_time'] if i+1<len(record['pulses']) else timestamps[-1]
            axis=int(pulse['direction'] in ('left','right'));sign=1 if pulse['direction'] in ('forward','left') else -1
            measured=sign*pulse['laser_body_delta'][axis]
            row={'file':str(filename),'slot':i,'direction':pulse['direction'],'measured_m':measured}
            for label,xy in trajectories.items():
                predicted=sign*(np.interp(b,times,xy[:,axis])-np.interp(a,times,xy[:,axis]))
                row[label+'_m']=float(predicted);row[label+'_error_m']=float(predicted-measured)
            results.append(row)
    if not results:raise ValueError('No validation pulses')
    metrics={label:{'rmse_m':math.sqrt(sum(r[label+'_error_m']**2 for r in results)/len(results)),
                    'max_abs_error_m':max(abs(r[label+'_error_m']) for r in results)} for label in trajectories}
    report={'run':args.run,'scenario_file':args.scenario_file,'n':len(results),'metrics':metrics,'results':results,
            'scope':'Held-out pure-axis settled displacement. Shared command ramp/wheel model and response replay; no collision scene or sensor metric-scale verification.'}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
