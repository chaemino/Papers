# Learning Disentangled Identifiers for Action-Customized Text-to-Image Generation

- 액션 맞춤형 텍스트-이미지 생성을 위한 분리된 식별자 학습
- 액션 관련 특징을 반전해 학습된 식별자 "<A>"를 다양한 캐릭터 및 동물과 짝을 이루어 정확하고 다양한 고품질 이미지 생성에 기여

<img width="677" alt="image" src="https://github.com/chaemino/Papers/assets/107089629/9f28bb0d-5c92-4f68-8737-4b3dae903951">

## Abstract

- 본 연구는 T2I(Text-to-Image) 생성의 새로운 작업, 즉 Action customization에 중점
- 작업의 목적은 제한된 데이터로부터 공존하는 액션(the co-existing action)을 학습하고 이를 인간이나 동물에게 일반화
- 실험 결과에 따르면 기존의 subject-driven customization methods는 액션의 대표적인 특성을 학습하지 못하고 **액션을 모양을 포함한 컨텍스트 기능에서 분리하는 데 어려움** 존재
  - subject-driven customization methods: 입력 이미지에서 중점이 되는 객체를 다양한 컨텍스트(상황, 배경, 소품 등)에 자연스럽게 융합하여 새로운 이미지를 생성하는 작업
- 낮은 수준의 특징에 대한 선호와 높은 수준의 특징의 얽힘을 극복하기 위해 예시 이미지로부터 액션별 식별자를 학습하는 반전 기반 방법인 ADI(Action Disentangled Identifier) 제안
  - 낮은 수준의 특징(low level appearance features): 이미지의 테두리, 텍스처를 의미하며 모델이 이를 우선적으로 학습하는 경향(선호)을 보임
    <-> 높은 수준의 특징: 특정 동작이나 개체의 형태
- 
