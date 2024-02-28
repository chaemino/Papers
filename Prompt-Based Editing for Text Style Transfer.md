# Prompt-Based Editing for Text Style Transfer

| 제목 |Prompt-Based Editing for Text Style Transfer |
|--|--|
| 저자 | Guoqing Luo and Yu Tong Han and Lili Mou and Mauajama Firdaus |
| 연도 |2023 |
| 학회/저널 | EMNLP|
| 논문 |[link](https://arxiv.org/pdf/2301.11997.pdf) |

- 발표 자료: [Paper_240221_PromptBasedEditing_TST_chaemin.pdf](https://github.com/chaemino/Papers/files/14427388/Paper_240221_PromptBasedEditing_TST_chaemin.pdf)
- 발표 영상: 

<br>

## Overview

1. Task(Text Styling Transfer(TST))
   - 목표: 문장을 한 스타일에서 다른 스타일로 변경하여 자동으로 다시 작성하는 것
   - 예시: "그는 샌드위치를 먹는 것을 좋아한다." → "그는 샌드위치를 먹는 것을 싫어한다."
   - 중요 포인트: 스타일을 변환하는 동안 문장의 스타일은 변경되어야 하지만 스타일과 독립적인 내용은 그대로 유지되어야 한다.
2. Prompting
   - 최근 PLM을 이용하여 다양한 자연어 생성 작업을 Zero-shot, Exampler-based 방식으로 수행
   - 해당 논문 역시 프롬프트 기반으로 학습 샘플이나 레이블 없이 PLM으로 직접 추론
3. Problem 
   - PLM으로 Query한 다음 Autoregressive 방식으로 스타일 변환된 문장 생성
   - 이러한 Autoregressive 방식을 이용한 생성흔 단어가 차례대로 생성되므로 결과에 대한 제어 어려움
   - 초기 오류가 향후 예측까지 영향을 주는 문제 발생 ***error accumulation problem**
4. Method
   - "Propose a Prompt-based Editing approach to unsupervised style transfer"
   - PLM 기반 Style Scorer 설계
     - PLM으로 Style 분류(Generation → Classification)하고 Classification Probability를 이용
     - 그 외 Fluency Score, Semantic Similarity Score 이용
   - Editing
     - 단어 수준(word-level)에서 editing 작업(치환, 삽입, 삭제)을 수행
     - top-k(k=50) 후보 문장 선정
   - Discrete Search: SAHC 수행
5. Advantage
   - 단어 단위로 문장 생성 대신 문장 전체 단어에 대한 editing을 수행하여 오류 누적 문제 해결
   - PLM 기반 Style Score를 유창성(fluency) 및 의미적 유사성(Semantic Similarity)과 같이 다른 점수 기능과 결합하는 개별 검색 알고리즘 설계
   - 보다 제어 가능하고(controllability) 정제된 문장 생성 가능


 
