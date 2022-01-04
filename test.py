from toolkit.toolkit_loader import  ToolKit_Loader

# print('runnn')
# if '':
#     print(True)
# sentimentor = toolkit_loader.ToolKit_Loader.load_sentiment_extractor(type = 1)
# with open('t.txt') as f:
#     text = f.read()
# import time
# st = time.time()
# for i in range(1):
#     print(sentimentor.sentiment_extract(topic = [],opinion = {'content': text}))

# e = time.time()
# print(e- st)

with open('t.txt') as f:
    text = f.read()

topic = ['macbook','review']
opinion = {'content': 'Xưa nay ko nghỉ đến việc mình xài macbook thay cho laptop Windows, nay ra chip M1, nên mạnh dạng đổi từ win sang Macbook Air. Cảm nhận ban đầu khá ngon về mọi thứ, bên win chưa có máy nào mà mình ko xài chuột ngoài, qua macbook bỏ hẳn chuột ngoài, thật vi diệu. Đặc biệt quả pin laptop thì khỏi nói, ko nóng xài sướng nữa '}

relation_extractor = ToolKit_Loader.load_relation_extractor(type = 1)
sentimentor = ToolKit_Loader.load_sentiment_extractor(type = 1)
print(sentimentor.sentiment_extract(topic = topic,opinion = opinion))
for i in range(1):
    print(relation_extractor.relation_extract(topic = topic,opinion = opinion))

