# Pokename Generator

description...

## Let's Generate Pokemon Name

```
$ poetry run python generate.py
Genereated:
ジキン
マイナ
チャーレ
ジキン
メガボスゴド
```

<img src="https://user-images.githubusercontent.com/17187586/109241944-938fed80-781d-11eb-8725-d38ef831b6a6.png" alt="sample_pokemon" style="width:500px">

## Dataset
- Japanese Dataset
  - https://rikapoke.com/pokemon_data8gen/
- English Dataset
  - https://www.kaggle.com/rounakbanik/pokemon


## Commands
### Train model
```
$ poetry run train.py
```

### Startup tensorboad
```
$ poetry run tensorboard --logdir=runs
```

## Reference
- [Seq2Seqを使ってポケモンの名前を作ったった - Qiita](https://qiita.com/yoyoyoyoyo/items/cfbbe8c65f9634763dec)
- [LSTMを用いた最強のポケモン生成 - Qiita](https://qiita.com/kntaaa000/items/93b9fe533857ff976037)
- [ニューラルネットワークでポケモンの名前＆特性を自動生成 - Gigazin](https://gigazine.net/news/20170404-pokemon-generated-neural-network/)
