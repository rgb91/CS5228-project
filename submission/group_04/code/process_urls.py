"""
    The script downloads the urls in parallel, and it uses the newspaper library to extract the text from
    the htmls. IT doesn't do any further processing till now.
"""
import re
import math
import threading
from newspaper import Article

num_threads = 6
non_alpha_regex = re.compile('[^a-zA-Z]')

def get_urls(urlpath):
    global num_threads
    urls = []
    with open(urlpath, "r") as fp:
        for line in fp:
           urls.append(line.strip().split("\t"))

    per_thread = int(math.ceil(len(urls)/num_threads))
    print(len(urls), per_thread, num_threads)
    for thread in range(num_threads):
        yield urls[thread*per_thread: (thread + 1)*per_thread]

def worker(thread_num, 
            urls, train_dir):
    url_text = []
    thread_file = open(train_dir + "Thread_"+str(thread_num) + ".dat", "a+")
    thread_file_key = open(train_dir + "Thread_keywords_"+str(thread_num) + ".dat", "a+")
    thread_file_summarized = open(train_dir + "Thread_summarized_"+str(thread_num) + ".dat", "a+")
    cnt = 0
    for (idx,url) in urls:
        print(thread_num, idx, url)
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            article.nlp()
            keywords = ','.join(article.keywords)
            summary = article.summary
            #print(keywords)
        except Exception as e:
            print("Error: " + str(e))
            thread_file.write("||".join([str(idx), "Empty"]))
            thread_file.write("###")
            thread_file_key.write("||".join([str(idx), "Empty"]))
            thread_file_key.write("###")
            thread_file_summarized.write("||".join([str(idx), "Empty"]))
            thread_file_summarized.write("###")
            continue
        thread_file.write("||".join([str(idx), text]))
        thread_file.write("###")
        thread_file_key.write("||".join([str(idx), keywords]))
        thread_file_key.write("###")
        thread_file_summarized.write("||".join([str(idx), summary]))
        thread_file_summarized.write("###")
        cnt += 1
        if cnt > 10:
            break

        print("Count: ", cnt)
    thread_file_key.close()
    thread_file_summarized.close()
    thread_file.close()


def scrape(url_fp, train_dir):
    threadn = 0
    threads = []
    for urls in get_urls(url_fp):
        t = threading.Thread(target=worker, args=(threadn, urls, train_dir))
        threads.append(t)
        t.start()
        print("Started Thread: ", str(threadn))
        threadn += 1
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    scrape("test_dir/test_url", "test_dir/")

