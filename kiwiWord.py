from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from konlpy.tag import Komoran

# Kiwi 객체 생성
kiwi = Kiwi()
komoran = Komoran()

# 텍스트를 형태소 분석하여 결과를 반환하는 함수
def analyze_text(text):
    result = kiwi.analyze(text)
    return result

# 형태소 분석 결과에서 명사를 추출하는 함수
def extract_nouns(text):
    nouns = []
    result = analyze_text(text)
    for token, pos, _, _ in result[0][0]:
        if len(token) != 1:# and (pos.startswith('N') or pos.startswith('SL')):
            nouns.append(token)
    return nouns

# 텍스트 예시
text = "안녕하세요. 저는 한국어 형태소 분석기인 Kiwi를 사용하여 명사를 추출하는 예제입니다."

# 명사 추출
# nouns = extract_nouns(text)
# print(nouns)
stopwords = Stopwords()
# print(kiwi.tokenize(text, stopwords=stopwords))
# print(kiwi.split_into_sents(text, return_tokens=True))
# print(komoran.pos(text))

f = open("./국무조정실_생소리_개선.txt", 'r', encoding='UTF8')
lines = f.readlines()
for line in lines:
    # print(line)
    result = komoran.pos(line)
    for token, pos in result:
        if len(token) != 1 and (
                pos.startswith('N') or pos.startswith('MM') or pos.startswith('MA') or pos.startswith('IC')):
            print(token, pos)

f.close()

