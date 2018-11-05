"""
    The script downloads the urls in parallel.
"""
import re
import threading
import urllib
from inscriptis import get_text

num_threads = 4
non_alpha_regex = re.compile('[^a-zA-Z]')

def get_urls(urlpath):
    global num_threads
    urls = []
    with open(urlpath, "r") as fp:
        for line in fp:
           urls.append(line.strip().split("\t"))

    per_thread = len(urls)/num_threads    
    for thread in range(num_threads):
        yield urls[thread*per_thread: (thread + 1)*per_thread]

def worker(thread_num, 
            urls):
    url_text = []
    thread_file = open(str(thread_num) + ".dat", "a+")
    for (idx,url) in urls:
        print(thread_num, idx, url)
        try:
            html = urllib.urlopen(url).read().decode("utf8")
            text = get_text(html)
        except Exception as e:
            print("Error: " + str(e))
            url_text.append((idx, ""))
            continue
        
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        chunks = [chunk for chunk in chunks if chunk]
        chunks = filter(lambda line: len(line.split(" ")) >= 8, chunks)
        #Remove lines with html tags
        chunks = filter(lambda word: "<" not in word, chunks)
        #Remove alphanumeric characters and then remove lines with html tags
        chunks  = map(lambda word: non_alpha_regex.sub(" ", word), chunks)
        thread_file.write("|".join([str(idx), "\n".join(chunks)]))
        thread_file.write("##")
    thread_file.close()

threadn = 0
threads = []
for urls in get_urls("urls"):
    t = threading.Thread(target=worker, args=(threadn, urls))
    threads.append(t)
    t.start()
    print("Started Thread: ", str(threadn))
    threadn += 1
