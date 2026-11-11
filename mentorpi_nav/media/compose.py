"""Compose the simulation demo video."""

def main():
    """Add explanatory overlays to captured simulation frames and encode a preview."""
    import json
    import math
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    out = ROOT / 'outputs/readme_demo'
    meta = json.loads((out / 'rollout.json').read_text())
    font_path = 'C:/Windows/Fonts/segoeui.ttf'
    bold_path = 'C:/Windows/Fonts/segoeuib.ttf'
    def font(size, bold=False):
        return ImageFont.truetype(bold_path if bold else font_path, size)

    frames = []
    writer = imageio.get_writer(out / 'mentorpi_goal_demo.mp4', fps=20, codec='libx264', quality=8,
                               macro_block_size=2, ffmpeg_params=['-movflags', '+faststart'])
    for i, r in enumerate(meta['records']):
        im = Image.new('RGB', (1280, 900), '#0c1423')
        d = ImageDraw.Draw(im)
        d.text((32, 18), 'MENTORPI', font=font(18, True), fill='#51decc')
        d.text((32, 45), 'From local map to goal.', font=font(36, True), fill='white')
        d.text((33, 95), 'Saved neural policy  /  Isaac Sim visualization  /  Analytic navigation dynamics', font=font(18), fill='#b2c1d8')
        scene = Image.open(out / f'scene_{i:04d}.png').convert('RGB').crop((180,0,1140,720)).resize((920, 690))
        im.paste(scene, (24, 138))
        d.rounded_rectangle((44, 156, 280, 197), radius=10, fill='#0c1423')
        d.text((58, 163), 'GOAL REACHED' if r['status']=='Goal reached' else 'POLICY IN CONTROL', font=font(18, True), fill='#51decc')
        d.text((974, 146), 'WHAT THE POLICY SEES', font=font(17, True), fill='white')
        grid = Image.open(out / f'grid_{i:04d}.png').convert('RGB').resize((260,260), Image.Resampling.NEAREST)
        im.paste(grid, (974, 184))
        d.ellipse((1099,309,1109,319), fill='#1688ff')
        d.line((1104,314,1104,296), fill='#1688ff', width=3)
        d.text((974, 458), '100 x 100 occupancy grid', font=font(18), fill='white')
        d.text((974, 488), 'Black: occupied  |  White: free', font=font(15), fill='#b2c1d8')
        d.text((974, 510), 'Blue marker: robot (overlay)', font=font(15), fill='#b2c1d8')
        d.text((974, 555), f"{r['distance']:.2f} m", font=font(43, True), fill='#51decc')
        d.text((974, 610), 'distance to goal', font=font(18), fill='#b2c1d8')
        d.text((974, 653), f"{r['time']:.1f} s", font=font(31, True), fill='white')
        d.text((974, 697), 'simulated time  |  2x playback', font=font(15), fill='#b2c1d8')
        d.text((974, 751), '7 obstacles | seed 789', font=font(17), fill='#b2c1d8')
        d.text((974, 781), 'Arrival radius: 0.15 m', font=font(17), fill='#b2c1d8')
        d.text((32, 848), '01  OBSERVE MAP     >     02  INFER MOTION     >     03  MOVE     >     04  REACH GOAL', font=font(19, True), fill='white')
        arr = np.asarray(im)
        copies = 30 if i == 0 else (50 if i == len(meta['records'])-1 else 1)
        for _ in range(copies):
            writer.append_data(arr)
        if i in (0, len(meta['records'])//2, len(meta['records'])-1):
            im.save(out / f'preview_{i:04d}.png')
        if i % 3 == 0 or i == len(meta['records'])-1:
            frames.append(im.resize((768,540)))
    writer.close()
    frames[0].save(out / 'mentorpi_goal_demo.gif', save_all=True, append_images=frames[1:],
                   duration=[1500]+[150]*(len(frames)-2)+[2500], loop=0)
    print(out / 'mentorpi_goal_demo.mp4')


if __name__ == "__main__":
    main()
