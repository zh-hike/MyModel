def my_model(args):
    config = dict(

        standard_method='MinMax',
        batch_size=999999,
        epochs=100,
        views_select=dict(
            voc=[0, 1],
        )
    )

    args.config = config
