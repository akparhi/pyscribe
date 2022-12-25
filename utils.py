import os
import time
import pytube
import re
import requests
from requests.exceptions import RequestException


def is_yt_url(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    return re.match(youtube_regex, url)


def get_filename(url):
    fragment_removed = url.split("#")[0]  # keep to left of first #
    query_string_removed = fragment_removed.split("?")[0]
    scheme_removed = query_string_removed.split("://")[-1].split(":")[-1]
    if scheme_removed.find("/") == -1:
        return ""
    return os.path.basename(scheme_removed)


def download_file(url, folder_name='files'):
    if (is_yt_url(url)):
        try:
            data = pytube.YouTube(url)
            fname = data.streams.get_audio_only().download(output_path=folder_name,
                                                           filename=str(time.perf_counter()) + '_' + data.streams[0].title + '.mp4')
            return '/'.join(fname.split('/')[-2:])
        except Exception as e:
            print(e)
    else:
        try:
            r = requests.get(url)
            fname = ''
            if "Content-Disposition" in r.headers.keys():
                fname = re.findall(
                    'filename="(.+)"', r.headers["Content-Disposition"])[0]
            else:
                fname = get_filename(url)

            saveto = folder_name + '/' + str(time.perf_counter()) + '_' + fname
            f = open(saveto, 'wb')
            f.write(r.content)
            return saveto
        except RequestException as e:
            print(e)
