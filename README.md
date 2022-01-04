# ml-toolkit

# Requirement

Các thư mục cần thiết có ở: 
https://drive.google.com/drive/u/1/folders/16UwM_8TOgZXw1IaUrSDeDH-XbjJMyIAV

/* Download các thư mục và đặt vào trong project*/

# Usage
```
from toolkit.toolkit_loader import  ToolKit_Loader

topic = ['macbook','review']
opinion = {'content': 'Xưa nay ko nghỉ đến việc mình xài macbook thay cho laptop Windows, nay ra chip M1, nên mạnh dạng đổi từ win sang Macbook Air. Cảm nhận ban đầu khá ngon về mọi thứ, bên win chưa có máy nào mà mình ko xài chuột ngoài, qua macbook bỏ hẳn chuột ngoài, thật vi diệu. Đặc biệt quả pin laptop thì khỏi nói, ko nóng xài sướng nữa '}

#Hiện tại chỉ có 1 model nên mặc định type = 1
relation_extractor = ToolKit_Loader.load_relation_extractor(type = 1) 
relation_result = relation_extractor.relation_extract(topic = topic,opinion = opinion)

sentimentor = ToolKit_Loader.load_sentiment_extractor(type = 1)
sentiment_result = sentimentor.sentiment_extract(topic = topic,opinion = opinion)

print(relation_result)
print(sentiment_result)
```

Result:
```
{'label': 'POS', 'score': 0.9976200461387634} #label can be "POS", "NEU" or "NEG"
{'score': 0.4379715992914262}
```

