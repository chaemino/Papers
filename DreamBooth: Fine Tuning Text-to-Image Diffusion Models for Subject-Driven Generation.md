# DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

_**"It’s like a photo booth, but once the subject is captured, it can be synthesized wherever your dreams take you..."**_

| 제목 | DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation |
|--|--|
| 저자 | Nataniel Ruiz and Yuanzhen Li and Varun Jampani and Yael Pritch and Michael Rubinstein and Kfir Aberman |
| 연도 |2023 |
| 학회/저널 | CVPR|
| 논문 |[link](https://arxiv.org/pdf/2301.11997.pdf) |

- 발표 자료: 
- 발표 영상: 

<br>

## Overview

- Subject 중심 이미지 생성을 위한 Text-to-Image Fine-Tuning 방법을 제안하는 논문이다.
  - Subject가 "개"인 경우, 아래와 같이 "원하는 개"(personalization)를 다양한 맥락(context)에서 재구성한 새로운 이미지 생성이 목표
    <img width="734" alt="image" src="https://github.com/chaemino/Papers/assets/107089629/93ee1e21-ae04-4d30-ac6e-4a651e93aa5d">
- Text-to-Image 모델은 AI 발전과 더불어 주어진 텍스트 프롬프트에서 고품질의 다양한 이미지 합성(systhesis)를 가능하게 만들었지만, 
여전히 주어진 reference set에서 subject의 모양을 모방하고, 다양한 맥락에서 새로운 표현을 합성하는 능력이 부족하다.
- 본 논문은 Text-to-Image Diffusion의 Personalization(사용자의 요구에 맞도록)에 대한 새로운 접근 방식을 제안한다.
- subject에 대한 몇 개의 이미지만 입력으로 주어지면 사전학습된 Text-to-Image 모델을 Fine-Tuning하여 해당 특정 주제와 unique identifier 방법을 학습한다.
- 
