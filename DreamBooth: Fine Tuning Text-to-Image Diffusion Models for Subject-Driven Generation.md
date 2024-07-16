# DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation


| 제목 | DreamBooth: Fine Tuning Text-to- Image Diffusion Models for Subject- Driven Generation |
|--|:--|
| 저자 | Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein and Kfir Aberman( Google Research ) |
| 연도 |2023 |
| 학회/저널 | CVPR|
| 논문 | [link](https://dreambooth.github.io/) |
| 발표 자료 | [pdf](https://velog.velcdn.com/images/chaemine/post/1da57634-d6ff-478d-bd81-bfd36a387d80/image.pdf) (연구실에서 진행한 논문 리뷰 자료를 공유합니다. 오류가 있을 시 편하게 연락부탁드립니다) |

_"It’s like a **photo booth**, but once the subject is captured, it can be synthesized wherever your dreams take you..."_




## Overview

- 주제 중심 이미지 생성을 위한 Diffusion Model Fine-Tuning
- Reference Set으로부터 주제의 fidelity(충실도)를 유지하면서 다양한 Context에서 표현
- Text-to-Image Diffusion Model의 Personalization을 위한 새로운 접근 방식 제안

### background

![](https://velog.velcdn.com/images/chaemine/post/92b699e1-d794-4721-8124-51d5b3080a87/image.png)

- 시계(subject)와 같은 특정 주제가 주어지면 text-image model을 사용하여 다양한 상황에서 해당 주제에 대한 높은 충실도를 유지하면서 이를 생성하는 것은 매우 어려운 작업
- 시계 모양에 대한 자세한 설명이 포함된 텍스트 프롬프트를 수십 번 반복하더라도 Imagen model(Saharia et al., 2022)은 주요 시각적 특징 재구성 불가
-   텍스트 임베딩이 공유된 language-vision space가 있고, 이미지의 의미론적 변형을 생성할 수 있는 모델인 DALL-E2(Ramesh et al., 2022)도 주어진 주제의 모양을 재구성하거나 수정 불가능
-   반면에 제안 방법은 시계를 높은 충실도와 새로운 맥락("a [V] clock in the jungle")으로 합성 가능

### method
- <span style='background-color:#fff51b'>Text-to-Image(T2I) Diffusion Models Fine-Tuning</span>
- <span style='background-color:#fff51b'>Personalization of T2I Models</span>
  - Designing Prompts: "[identifier] [class noun]"
  - Rare-token Identifiers
    - "How to get **unique identifier?**"
  - 모델의 입력으로 프롬프트를 주되, identifier를 추가적으로 제공
- <span style='background-color:#fff51b'>class-specific Prior Preservation loss</span>
  - identifier를 추가적으로 제공한 class noun(생성하고자 하는 대상)이 아닌, class noun에 대한 보편적인 이미지 생성을 유지하기 위해 추가적인 Loss 사용


---

## Abstract

- Large text-to-image(T2I) 모델은 AI 발전으로 주어진 텍스트 프롬프트에서 고품질의 다양한 이미지 합성을 가능하게 만듦
- 그러나 이러한 모델은 주어진 reference set에서 subjects 모양을 모방하고 다양한 맥락(context)에서 새로운 표현을 합성하는 능력 부족
- 본 논문은 **T2I Diffusion model의 personalization에 대한 새로운 접근 방식** 제안
- 주제에 대한 몇 개의 이미지만 입력으로 주어지면 사전 학습된 T2I 모델을 Fine-Tuning하여 해당 특정 subject와 고유 식별자(unique identifier) 바인딩 방법 학습
- subject가 output domain에 포함이 되면 unique identifier를 사용하여 다양한 장면에서 상황에 맞는 subject의 완전히 새로운 사실적 이미지 합성 가능
- 이전에는 해결할 수 없었던 여러 작업에 제안 기술 적용 가능
    - 주제 재구성(subject reconstruction)
    - 텍스트 기반 합성(text-guided view synthesis)
    - 모양 수정(appearance modification)
    - 예술적 랜더링(artistic rendering)
    
<br>

## Introduction

|![](https://velog.velcdn.com/images/chaemine/post/32977af4-a996-4bc7-9a31-f0cdb7aff03e/image.png) |
|--|
|- 피사체(sunject)에 대한 몇 개의 이미지(일반적으로 3-5개) 만으로 텍스트 프롬프트를 사용하여 다양한 상황(context)에서 피사체의 이미지 생성 가능 <br> - 실험 결과, 주위 환경과의 자연스러운 상호 작용 뿐만 아니라 새로운 표현과 조명 조건의 변화를 보여주고 동시에 피사체의 주요 시각적 특징에 대한 높은 충실도(fidelity) 유지 |


- 상상의 장면을 랜더링 하고, 원할하게 장면에 혼합되도록 새로운 상황(context)에서 특정 subject(위의 이미지에서는 "개")를 합성해야 하는 작업은 어려움
- 최근 개발된 대규모 T2I 모델은 자연어로 작성된 텍스트 프롬프트를 기반으로 고품질의 다양한 이미지 합성을 가능하도록 함
- 이러한 모델의 주요 장점은 대규모 image-caption 쌍 모음에서 사전에 학습된 강력한 의미 체계(the strong semantic)
- 사전 학습은 이미지에서 다양한 포즈와 상황으로 나타날 수 있는 개의 다양한 사례와 "개"라는 단어를 연결하는 방법 학습 가능
    - 예를 들어, [Figure 1.]()에서 "in the Acropolis"와 같은 상황에서 표현
- 주어진 reference set에서 subject의 모양을 모방하고, 다른 맥락에서 동일한 subject의 모습을 정확하게 재구성 불가능
    - instance에 대한 자세한 텍스트 설명조차도 모양이 다른 instance를 생성
- 또한 공유된 language-vision space에 text embedding이 있는 모델도 주어진 subject의 모습을 정확하게 재구성 불가능
- 이 논문에서는 T2I Diffusion Model의 개인화(Personalization)를 위한 새로운 접근 방식 제안

### In this paper

#### [method 1] Personalization of Text-to-Image Models

- **목표**: 사용자가 생성하려는 특정 주제(subject)에 새로운 단어를 연결하도록 모델의 language vision 사전 확장

- 새로운 단어가 모델에 내장되면 이러한 단어를 사용하여 주요 식별 기능(key identifying features)을 유지하면서 다양한 장면(context)에서 상황에 맞게 주제에 대한 새로운 사실적 이미지를 합성 → 이 효과는 "magic photo booth"와 유사
- subject의 이미지를 몇 장 촬영하면 부스는 **간단하고 직관적인 텍스트 프롬프트에 따라** 다양한 조건과 장면에 subject 사진 생성
- 공식적으로 **subject에 대한 몇 개의 이미지(3-5개)가 주어지면** subject를 모델의 output domain에 이식하여 **고유한 식별자(unique identifier)로 합성 가능**하도록 하는 것이 목표
- 이를 위해 저자들은 희귀한 토큰 식별자로 주제를 표현하고 사전 학습된 Diffusion 기반 T2I framework를 Fine-Tuning

- **결론**: unique identifier와 subject의 class name(e.g., "A \[V] dog")이 포함된 입력 이미지와 텍스트 프롬프트를 사용하여 Fine-Tuning

> **[method 1]** 본인이 키우는 강아지 사진 몇 장을 가지고 강아지가 수영하고, 관광을 간 사진을 생성하고 싶은데 그러기 위해서 단순히 텍스트 프롬프트로 강아지(class name)를 제공하는 것이 아니라, "\[V]"와 같이 unique identifier를 사용해서 구별을 하여 Fine-Tuning 한다는 방식이라고 보면 됩니다.


#### [method 2] Class-specific Prior Preservation Loss

- subject의 class name을 사용하면 모델이 subject가 되는 class에 대한 사전 지식을 사용하는 동시에 클래스별 인스턴스(class-specific instance)가 unique indentifier와 바인딩될 수 있음
- **language drift를 방지하기 위해** 모델에 포함된 의미(semantic prior)를 활용하고, **subject와 동일한 클래스의 다양한 instance를 생성하도록 장려하는 클래스별 사전 보존 손실(class-specific prior preservation loss) 제안**

> **language drift**
> - 언어 모델이 대규모 데이터셋으로 사전 학습된 이후 downstream task에 대해 Fine-Tuning 되면 언어에 대한 구문 및 의미론적 지식을 점진적으로 잃음
- **모델은 대상 주제와 동일한 class의 주제를 생성하는 방법을 점지적으로 잃음**


> **[method 2]** "A \[V] dog"로 Fine-Tuning을 수행하였을 때, class name에 해당하는 dog에 대하여 "A dog"로 이미지를 생성하면 평범한 dogs의 이미지가 아닌 "\[V] dog"만 생성되었다는 뜻입니다. 저자들은 unique한 identifier 없이 class name에 대한 사전 지식을 유지하고 싶어서 따로 class name에 대한 이미지를 샘플링하여 생성하고, 이를 이용하여 loss를 계산해서 더해주는 방법을 제안합니다.


<br>

## Related work

### Image Compostion

![](https://velog.velcdn.com/images/chaemine/post/1c4adff7-e304-40db-9994-1a01bcb5a0d6/image.png)


- Image composition techniques는 주어진 subject를 새로운 배경으로 복제하여 subject가 장면에 융합되도록 하는 것이 목표
- 새로운 포즈의 구성을 고려하기 위해 일반적으로 단단한(rigid) 물체에 작동하고 더 많은 수의 뷰가 필요한 3D 재구성 기술 적용 가능
- 일부 단점으로는 장면 통합(scene integration)(조명, 그림자, 접촉)과 새로운 장면을 생성할 수 없다는 점 등 존재
- **이와 대조적으로 이 논문의 제안 방법은 새로운 포즈와 상황에 subject를 합성 및 자연스러운 생성 가능**

### T2I Editing and Synthesis

![](https://velog.velcdn.com/images/chaemine/post/41dc8cb0-9393-4f09-99f7-8d94e1eb5e90/image.png)
 

- 텍스트 기반 이미지 조작은 최근 텍스트를 사용하여 사실적인 조작(manipulations)을 제공하는 CLIP 과 같은 Image-Text 표현과 결합된 GAN을 사용하여 상당한 진전을 이룸
- 이러한 방법은 구조화된 시나리오(e.g., human face editing)에서 잘 작동하지만 주제가 다양한 데이터셋에서는 어려움 존재
- 다른 연구는 매우 다양한 데이터셋에 대해 최첨단 생성 품질을 달성하고 종종 GAN을 능가하는 **Diffusion Model** 사용
- 편집 접근 방식의 대부분은 특정 이미지의 전역 속성 수정이나 로컬 편집을 허용
- 그러나 새로운 맥락에서 특정 주제에 대한 새로운 표현 생성 불가능 
→ **T2I 합성(synthesis)** 작업 존재
- T2I 합성에서 Imagen, DALL-E2, Parti, CogView 및 Stable Diffustion은 최근의 large text-to-image 모델은 전례 없는 생성 능력 증명
- 그러나 여전히 합성된 이미지 전체에서 대상의 정체성을 일관되게 유지하는 것은 어렵거나 불가능


<br>

## Problems

![](https://velog.velcdn.com/images/chaemine/post/3f27ff00-4750-4bdd-89fc-86734c60e150/image.png)


→ 주제(subject, e.g., retro style yellow alarm clock)가 주어졌을 때 Text-to-Image Model을 사용하여 다양한 맥락(context)에서 해당 주제에 대한 높은 충실도(fidelity)를 유지하면서 이미지를 생성하는 어려움


<br>

## Architecture

- subject의 3-5개의 이미지가 주어지면 2 step으로 T2I Diffusion Model Fine-Tuning

![](https://velog.velcdn.com/images/chaemine/post/7b07592f-6f89-47dc-982d-4a2b752aa621/image.png)

1. Text-to-Image Model Fine-Tuning using "Text-Prompt" and "Input Images"
2. Super-Resolution Fine-Tuning
    - SR Module을 Fine-Tuning하여 subject의 디테일을 보존하고 Phorealistic하게 만듦

<br>

## Method

###  1. Personalization of Text-to-Image Models

- 첫 번째 작업은 subject instance를 모델의 output domain으로 이식하여 모델에 subject의 다양하고 새로운 이미지를 query 할 수 있도록 하는 것
- 자연스러운 아이디어 중 하나는 subject의 몇 장의 데이터셋을 이용하여 모델을 FIne-Tuning

> GAN과 같은 Generative model은 과적합 및 mode-collapse를 유발할 수 있을 뿐만 아니 라 대상 분포를 충분히 잘 포착(capturing)하지 못할 수도 있으므로 주의를 기울이 필요성 존재하였고 이러한 함정을 피하기 위한 기술에 대한 연구 진행 → 그러나 제안 방법과 달리 주로 대상 분포와 유사한 이미지를 생성하나 subject 보존에 대한 요구사항은 없음

- 우리는 Diffusion Loss를 사용하여 Fine-Tuning을 수행하면 Large text-to-image diffusion 모델이 이전 정보를 잊거나 작은 학습데이터셋에 과적합되지 않고 새로운 정보를 domain에 통합하는 데 효과를 보이는 특이한(peculiar) 발견 관찰

#### 1.1. Designing Prompts for Few-Shot Personalization

- 목표는 Diffusion model의 "사전(dictionary)"에 새로운(unique identifier, subject)쌍을 "implant(삽입)" 하는 것
- 주어진 이미지 세트에 대한 자세한 이미지 설명을 작성하는 오버헤드(overhead)를 우회하기 위해 더 간단한 접근 방식 선택
- 주제의 모든 입력 이미지에 **"[identifier] [class noun]"**이라는 레이블 지정
  - 여기서 **[indentifier]**는 주제에 연결된 고유 식별자(unique identifier)이고,
  - **[class noun]**은 주제의 대략적인 클래스 설명자(e.g., cat, dog, watch, etc.)
- class descriptor(**[class noun]**)는 사용자가 제공하거나 분류기(classifier)를 사용하여 얻을 수 있음
- class의 이전 항목을 unique subject에 묶기 위해 문장에 class descriptor를 사용하고 잘못된 class descriptor를 사용하거나 class descriptor가 없으면 학습 시간과 language drift가 증가하고 성능 저하 확인


   - | [class noun]이 없고 무작위로 샘플링된 잘못된 [class noun]이 있는 데이터셋으로 실험 |
    |--|
    | ![](https://velog.velcdn.com/images/chaemine/post/3513f212-5d7a-49b7-894e-00d820cf664c/image.png)|
    

#### 1.2. Rare-token Identifiers

-   일반적으로 기존 영어 단어(예: "unique", "special")가 차선책이라고 생각
-   모델은 원래 의미에서 단어를 풀고 주제를 참조하기 위해 다시 얽히는 방법을 배워야 하고
-   이는 언어 모델과 diffusion 모델 모두에서 약한 사전성(prior)을 갖는 식별자(identifier)에 대한 필요성 유발
-   이를 수행하는 위험한 방법은 영어에서 임의의 문자를 선택하고 이를 연결하여 희귀한 식별자("xxy5syt00") 생성
-   실제로 토크나이저는 각 문자를 개별적으로 토큰화할 수 있으며, diffusion 모델의 prior는 이러한 문자에 대해 강력
-   이러한 토큰이 일반적인 영어 단어를 사용하는 것과 유사한 약점을 갖는 것을 종종 발견
-   접근 방식은 **어휘에서 희귀한 토큰을 찾은 다음 이러한 토큰을 텍스트 공간(vector space)으로 반전시켜 식별자가 강력한 사전 확률을 가질 확률을 최소화**
-   어휘에서 희귀 토큰 조회를 수행하고 일련의 희귀 토큰 식별자 $f(\hat{V})$​을 얻음($f$는 토크나이저)
-   $f$는 문자 시퀀스를 토큰에 매핑하는 함수이고, $\hat{V}$는 토큰 $f(\hat{V})$에서 유래하는 디코딩된 텍스트
-   시퀀스는 가변 길이 $k$일 수 있으며, $k = \{1, ..., 3\}$​의 상대적으로 짧은 시퀀스가 잘 작동함을 알 수 있음
-   그런 다음 $f(\hat{V})$의 detokenizer를 사용하여 어휘를 반전함으로 unique identifer $\hat{V}$를 정의하는 문자 시퀀스를 얻음
-   Imagen의 경우 3개 이하의 유니코드 문자(공백 없이)에 해당하는 토큰의 균일한 무작위 샘플링을 사용하고, T5-XXL 토크나이저 범위는 $\{5000, ..., 10000\}$의 토큰을 사용하는 것이 효과적이라는 것을 확인


#### 1.3. 추가적인 설명

- 간단히 말해서, 생성하고자 하는 subject가 "container"라고 할 때 생성하고 싶은 건 모델의 입력으로 제공할 특정한 container이지, 보편적인 container 이미지가 아닙니다.
- 그래서 Text-Prompt로 "a photo of a container"가 아니라, unique identifier로 희귀 토큰을 찾아서 넣습니다.
- unique identifier는 잘 사용하지 않는 토큰을 찾아서 사용합니다.

![](https://velog.velcdn.com/images/chaemine/post/ad9cd8aa-947e-4c97-8196-f5d15d3d43b3/image.png)

<br>

### 2. Class-specific Prior Preservation Loss

- 단순히 주제의 입력 이미지와 텍스트 프롬프트(“[identifier] [class noun]”)로 얻어진 condintioning vector $c_s$ 에 대해 기존 loss로 Fine-Tuning → **Overfitting, Language Drift** 발생

| overfitting | language drift |
|--|--|
| ![](https://velog.velcdn.com/images/chaemine/post/2db310de-ded5-4ecc-91f8-1d16109f09a5/image.png)| ![](https://velog.velcdn.com/images/chaemine/post/6f22e4eb-516d-49ef-bd90-684dccc9ea44/image.png)

- overfitting
    - 주어진 입력 데이터셋이 적어서 overfitting이 될 가능성 존재
    - Fine-Tuning 시 subject-fidelity와 semantic modification의 균형 불확실
    → Model의 모든 레이어를 Fine-Tuning하면 subject-fidelity를 극대화하는 최상의 결과 확인
    그러나, language drift 발생
    
- language drift
    - 언어 모델이 대규모 데이터셋으로 사전 학습된 이후 downstream task에 대해 Fine-Tuning되면 언어에 대한 구문 및 의미론적 지식을 점진적으로 잃음
    - **모델은 대상 주제와 동일한 class의 주제를 생성하는 방법을 점진적으로 잃음**
    (동일한 class에 속하는 다른 instance를 생성하지 못함)
    
- ![](https://velog.velcdn.com/images/chaemine/post/d92070ca-30c4-427f-afac-90a5c3e127e9/image.png)


- 따라서, 단순히 class noun에 대한 이미지를 여러 장 생성하고 이를 이용하여 발생한 loss를 기존의 loss에 더해주는 방식
- 단순함에도 불구하고 이러한 prior preservation loss가 앞선 문제를 극복하는 데 효과적임을 발견
- 이 과정에서 개의 “a [class noun]” 샘플이 생성되지만 그보다 적은 양 사용 가능

#### source code 

```python
 if self.args.with_prior_preservation: # with prior preservation -> class name 샘플링 사용
# Chunk the noise and model_pred into two parts and compute the loss on each part separately.
	model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)
    # Compute prior loss
    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
    
(생략)
if self.args.with_prior_preservation:
# Add the prior loss to the instance loss.
	loss = loss + self.args.prior_loss_weight * prior_loss # 설정한 가중치 만큼 더해줌

```

- [soruce code](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)


### 3. Personalized Instance-Specific Super-Resolution

- SR(Super-Resoulution) 모델들은 이미지에서 주제의 디테일을 보존하고 사실적으로 만듦
- 그렇지 않으면 생성된 이미지에서 부정확한 feature 묘사, 디테일 상실 등이 발생
- 64X64 → 256X256 Fine-Tuning은 필수 
- 256X256 → 1024X1024 모델을 Fine-Tuning하면 높은 수준의 세밀한 디테일을 가지는 데 도움

![](https://velog.velcdn.com/images/chaemine/post/fba61cd5-194f-400f-bac0-db4903742e7e/image.png)

<br>

## Experiments


### 1. DataSet

- 30개의 subject 데이터셋 수집(배낭, 인형, 개, 고양이, 선글라스, 만화, 물체, 반려동물 등)
- 데이터셋의 이미지는 작성자가 수집했거나 [Unsplash](https://unsplash.com/)에서 다운로드
- 평가의 경우 30개의 주제별 25개의 프롬프트 마다 4개의 이미지 생성, 총 3000개의 이미지 생성

### 2. Evaluation Metrics

- CLIP-I
    - 테스트 이미지들의 CLIP Similarity 평균값
    - 생성된 이미지와 실제 이미지의 CLIP Embedding 간의 Cosine Similarity
    - 유사한 텍스트 설명(e.g., two different yellow clocks)을 가질 수 있는 서로 다른 주제를 구별하지 못하는 한계점 존재
    
- DINO
    - CLIP과 달리 동일 class의 instance 간의 차이를 무시하지 않도록 학습
    - self-supervised 방식으로 학습이 되어 주제 또는 이미지의 unique feature 구별 가능
    
- CLIP-T
    - 프롬프트 fidelity를 측정하기 위해 텍스트 프롬프트 및 이미지 CLIP Embedding 간의 평균 Cosine Similarity 비교
    
### 3. Comparisons

- Imagen/Stable Diffusion을 사용하여 DreamBooth 이미지 생성
- Textual Inversion(Gal et al.)과 비교
- Textual Inversion(Gal et al.)에 비해 DreamBooth에 대한 주제 및 프롬프트 충실도 지표 모두 높은 점수

![](https://velog.velcdn.com/images/chaemine/post/5543f618-5a59-4bd8-9cce-28d051bac089/image.png)

### 4. Applications

#### 4.1. Recontextualization

- 설명 프롬프트(Descriptive Prompt)(“a [V] [class noun] [class description]”)를 사용하여 다양한 맥락에서 특정 주제에 대한 새로운 이미지 생성 가능
- 이전에는 볼 수 없었던 구조와 장면 내 주제의 사실적인 통합으로 새로운 이미지 생성 가능

![](https://velog.velcdn.com/images/chaemine/post/1beed5c0-a0de-4a87-91d5-d7b23e9d23a8/image.png)


#### 4.2. Art Renditions

- **"a painting of a [V] [class coun] in the style of [famous painter]"** 또는 **"a statue of a [V] [class noun] in the style of [famous sculptor]"**라는 프롬프트가 주어지면 subject에 대한 예술적 표현 생성 가능

![](https://velog.velcdn.com/images/chaemine/post/f2fd630d-12d1-430e-a395-7fa44dbbe90f/image.png)


## Conclusion

- subject에 대한 몇 가지 이미지와 텍스트 프롬프트로 다양한 맥락에서 subject에 대한 이미지 생성 가능
- 핵심 아이디어는 subject를 unique indentifier에 바인딩하여 Text-to-Image Diffusion Model의 출력 도메인에 특정 subject Instance를 적용하는 것
- Fine-Tuning 과정은 3~5개의 subject 이미지에서만 작동 가능하므로 쉽게 기술 사용 가능
- 그러나 여전히 subject의 복잡성에 따라서 학습 시간의 차이나 Overfitting 존재


