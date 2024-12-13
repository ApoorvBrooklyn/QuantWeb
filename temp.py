import ycnbc

news = ycnbc.News()
#getting latest news
latest = news.latest()
print(latest)