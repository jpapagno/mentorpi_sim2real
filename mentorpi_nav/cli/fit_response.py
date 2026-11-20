"""Fit an experimental yaw-history model against timestamped bag exports.

Inputs are read-only exports, not live robot topics. Separate takes must be
reserved for validation. No profile is deployed or activated by this command.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--signals',required=True)
    p.add_argument('--records',nargs='+',required=True)
    p.add_argument('--fit-takes',nargs='+',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    import numpy as np
    from scipy.optimize import differential_evolution
    from mentorpi_nav.drivetrain_response import YawResponse,YawResponseCfg
    records={Path(path).stem.rsplit('_',1)[1]:json.loads(Path(path).read_text()) for path in args.records}
    signals=json.loads(Path(args.signals).read_text());prepared=[];dt=.02
    for signal in signals:
        take=signal['take'];record=records[take];guards=signal['topics']['/mentorpi/guard_status']
        offset=min(row['receipt']-row['guard']['last_enable']['at'] for row in guards)
        gt=np.array([row['receipt']-offset for row in guards]);u=np.array([row['guard']['applied'][2] for row in guards])
        active=np.flatnonzero(abs(u)>.005)
        t=np.arange(gt[active[0]]-.4,gt[active[-1]]+1.,dt)
        command=u[np.searchsorted(gt,t,side='right').clip(1,len(gt))-1]
        imu=signal['topics']['/imu'];it=np.array([r['stamp']-offset for r in imu]);w=np.array([r['gyro_z'] for r in imu])-record['stationary_gyro_z_bias_radps']
        measured=np.interp(t,it,w)
        prepared.append({'take':take,'t':t,'u':command,'measured':measured,'record':record})
    fitting=[r for r in prepared if r['take'] in args.fit_takes];validation=[r for r in prepared if r['take'] not in args.fit_takes]
    if not fitting or not validation:p.error('Both fitting and validation takes are required')
    def replay(row,parameters):
        cfg=YawResponseCfg(breakaway_memory=parameters[0],rise_time_s=parameters[1],rate_gain=parameters[2],stop_time_s=parameters[3])
        model=YawResponse(cfg)
        rate=np.array([model.step(float(v),dt) for v in row['u']])
        angle=np.r_[0,np.cumsum((rate[1:]+rate[:-1])*.5*dt)]
        return rate,angle
    def objective(parameters):
        errors=[]
        for row in fitting:
            rate,angle=replay(row,parameters)
            error=rate-row['measured']
            errors.append(float(np.mean(np.minimum(error*error,.04))))
        return sum(errors)/len(errors)
    fit=differential_evolution(objective,[(.005,.18),(.015,.6),(.6,1.4),(.008,.15)],
                               seed=72002,popsize=8,maxiter=55,tol=.003,polish=False,workers=1)
    cfg=YawResponseCfg(breakaway_memory=float(fit.x[0]),rise_time_s=float(fit.x[1]),rate_gain=float(fit.x[2]),stop_time_s=float(fit.x[3]))
    report={'version':'yaw_history_response_v1','status':'experimental_replay_only','parameters':asdict(cfg),
        'fit_takes':args.fit_takes,'validation_takes':[r['take'] for r in validation],
        'source_sha256':{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [args.signals,*args.records]},
        'assumptions':['cold memory state at start of each capture','10second idle memory decay fixed, not identified',
                       'minimum command .205rad/s provisional, not fitted on these .3rad/s takes',
                       'signed drive memory is phenomenological; does not identify board firmware mechanism'],
        'scope':'pure yaw at0.3rad/s on one surface; not validated for mixed motion or other voltages',
        'results':[]}
    plots=[]
    for row in prepared:
        rate,angle=replay(row,fit.x);measured_angle=np.r_[0,np.cumsum((row['measured'][1:]+row['measured'][:-1])*.5*dt)]
        result={'take':row['take'],'split':'fit' if row['take'] in args.fit_takes else 'validation',
                'rate_rmse_radps':float(np.sqrt(np.mean((rate-row['measured'])**2))),'pulses':[]}
        for pulse in row['record']['pulses']:
            start=pulse['start_time'];end=pulse['command_end_time']+1.
            predicted=float(np.interp(end,row['t'],angle)-np.interp(start,row['t'],angle))
            measured=float(np.interp(end,row['t'],measured_angle)-np.interp(start,row['t'],measured_angle))
            ideal=float(np.sum(row['u'][(row['t']>=start)&(row['t']<=end)])*dt)
            result['pulses'].append({'direction':pulse['direction'],'predicted_angle_degrees':float(np.degrees(predicted)),
                'measured_gyro_angle_degrees':float(np.degrees(measured)),
                'error_degrees':float(np.degrees(predicted-measured)),
                'ideal_command_angle_degrees':float(np.degrees(ideal))})
        report['results'].append(result);plots.append((row,rate))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'yaw_history_profile.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(plots),1,figsize=(11,10),layout='constrained')
    for ax,(row,rate),result in zip(axes,plots,report['results']):
        t=row['t']-row['t'][0]
        ax.plot(t,row['u'],label='Applied command',alpha=.6)
        ax.plot(t,row['measured'],label='Measured gyro',alpha=.7)
        ax.plot(t,rate,label='Stateful replay',linewidth=2)
        ax.set_title('Take '+row['take']+' — '+result['split']);ax.set_ylabel('rad/s');ax.grid(alpha=.2)
    axes[0].legend(loc='upper right');axes[-1].set_xlabel('Seconds from replay start')
    fig.suptitle('History-dependent yaw response: fit and separate mirrored validation takes')
    fig.savefig(out/'yaw_history_validation.png',dpi=140)
    print(json.dumps({'parameters':asdict(cfg),'results':report['results']},indent=2),flush=True)


if __name__=='__main__':main()
