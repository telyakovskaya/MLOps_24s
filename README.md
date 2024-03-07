# MLOps_24s
# 1. Формулировка задачи
Наконец пришла весна, поэтому, соответствуя настроению возрождения природы, возьмем задачу распознавания цветов по фото.

На нашей планете огромное количество разнообразных видов цветов. Некоторые, например, розы, имеют множество оттенков. Названия и детали каждого цветка трудно запомнить. Кроме того, люди могут ошибочно идентифицировать похожие виды цветов.

В настоящее время единственным способом идентифицировать какой-либо конкретный цветок или его разновидности является поиск информации, основанный на собственных знаниях и профессиональном опыте. Наличие таких знаний не всегда возможно. Сегодня единственными реальными вариантами поиска такого контента в Интернете являются поиск по ключевым словам и текстовые редакторы. Проблема в том, что даже в этом случае поисковику все равно нужно будет подобрать подходящие релевантные ключевые слова, чего он сделать не в состоянии.

Рассмотрим, как использовать нейросети для распознования цветов и в начале возьмем всего 5 видов.

Итак, для каждой картинки нужно определить один из пяти типов цветов. Задача не очень сложная, но кажется, как раз подойдет для курса по MLOps)

# 2. Данные
Возьмем, как было предложено, датасет с kaggle: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition

Фото разделены на пять классов: ромашка, тюльпан, роза, подсолнух, одуванчик.
Для каждого класса представлено около 800 фотографий. Фотографии имеют невысокое разрешение (около 320x240 пикселей). Они не сводятся до единого размера, то есть имеют разные пропорции (с этим могут быть нюансы, поэтому стоит обратить на этот факт отдельное внимание).

Всего фотографий в датасете 4242.
Датасет основан на собранных данных с flickr, Google images и yandex images.
Возможно, данных окажется слишком много, поэтому придется и порезать, но не думаю, что это станет проблемой.

![alt text]([http://url/to/img.png](https://www.kaggleusercontent.com/kf/65532056/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..BDQ7sPgbeszruw8cMrx2YQ.NrLln-Ba8H7WOTtXLGasLXABPz0VruUhJNBfPigknfDBzkXSpbUd3VXHzZwQAWdIMXntEXoEAz9PKeLlEIW8VcRPdrkGyNlK8SRQ44PEuYh44ZzjGs811ZJrkThPvWaka1T_ZFwjI-Yt-gf4NoySLpQcNskLBUWUVJ9pGO3_P97Ga4thgJirKDnl2FqgxTrNPlfPR_ue9DCSEvXWP_J1uk7GrPUAihSsBsY0cPU8DriU6OBqTVAY4uAnHaT9fwGPSV5FDFPA_iQcQplofAs8yfmnW891Vg03Cz10yUkYX2Ecmrla_Q62iwo_NCjsqnpZsVf9wHzz52IJVWgD8Q9TIMHbAc16zROEPn_BymEvrtLldRy_zpCSZMHvsKUT7HWKa05wLyadITmt3ZZBWcLihlhSocq0FSWyBkoKlbO2-I1UuX-4b48zOUNeNk92YOFak8b9gIKd_gaah4m0VXZT2jBW8AClHqj7S_aqDsGa-RU3fyqoqoozbhBY3b712Szel3LN5OtnOpEVsQfVCXzc_OrsN7n7cNc4IVHqIvhIoFk2Ew_tXmQb-uyfS23vMbbebdbysvA0RT2KTpjtYHzqZotoQAsDBWbyLK93YWUwhISCBfyjMvtqPb1OgewPHeJblRRJAhCzarLO7brVoNac66yGd70H5XVgR9FFw7e9MKGlcqh0tjXx5il376dX-RvFt9JRV7QWsffWAljB8_Ej0A.1UpGFbjEpdqSTZlhia9HsA/__results___files/__results___30_0.png))
# 3. Подход к моделированию
Будем решать данную задачу с помощью нейросетей трансформеров и библиотеки pytorch и transform. На тренировочной части датасета обучится пять широко известных моделей: DenseNet(1), DenseNet(2), GoogleNet, ResNet и VGG19. Потом с помощью ансамблирования обученных моделей получим предсказание для тестовой части выборки. Конечно, перед обучением нужно будет сделать предобработку картинок в силу специфики датасета и неодинаковости его элементов. За основу решения возьмем ноутбук с kaggle с очень высокой точностью решения данной задачи: https://www.kaggle.com/code/georgiisirotenko/pytorch-flowers-translearing-ensemble-test-99-67/notebook
# 4. Способ предсказания
Итак, шаги для Production Pipeline (CI): Data loading, data preparation, model training, model evaluation и model validation. Конечно, серединные шаги будут выполняться для кажой из пяти моделей, а потом уже применяться ensembling.
