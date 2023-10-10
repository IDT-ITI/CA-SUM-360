import ffmpeg

input_360_file = 'CA-SUM-360/scripts/360video.mp4'
output_ERP_file = 'CA-SUM-360/scripts/erpvideo.mp4'

try:
    ffmpeg.input(input_360_file).output(output_ERP_file, vf="v360=eac:equirect").run(overwrite_output=True, capture_stderr=True)
except ffmpeg.Error as e:
    print('Error:', e.stderr.decode())
