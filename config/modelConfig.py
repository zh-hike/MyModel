def my_model(args):
    hidden_dim = args.hidden_dim
    config = dict(



        epochs=200,
        pre_epochs=200,
        views_select=dict(
            voc=[0, 1],
            mnist=[0, 1],
        ),
        needpretrain=True,

        network=dict(
            voc=dict(
                standard_method='MinMax',
                batch_size=9999999,
                seed=5,
                n_classes=20,
                cluster=dict(
                    dims=[hidden_dim, 64, 20],
                    lr=1e-3,
                    batchnorm=True,
                    activate='ReLU',
                    out_activate='Softmax',
                ),
                attention=dict(
                    tau=10,  # attention的温度参数
                    dims=[hidden_dim*2, hidden_dim, 64, 2],
                    lr=1e-4,
                    batchnorm=True,
                    activate='LeakyReLU',
                    out_activate='Sigmoid',
                ),
                autoencoder=dict(
                    lr=1e-3,
                    hidden_dim=hidden_dim,
                    encoder=dict(
                        encs=[[512, 1024],
                              [399, 1024], ],

                        batchnorm=False,
                        activate='ReLU',
                        out_activate='ReLU',
                    ),
                    decoder=dict(
                        decs=[[1024, 512],
                              [1024, 399]],
                        batchnorm=False,
                        activate='ReLU',
                        out_activate=None,
                    ),
                )

            ),
            mnist=dict(
                standard_method='MinMax',
                batch_size=9999999,
                seed=18,     # 3:88
                n_classes=10,
                cluster=dict(
                    dims=[hidden_dim, 64, 10],
                    lr=1e-3,
                    batchnorm=True,
                    activate='ReLU',
                    out_activate='Softmax',
                ),
                attention=dict(
                    tau=10,  # attention的温度参数
                    dims=[hidden_dim * 2, hidden_dim, 64, 2],
                    lr=1e-4,
                    batchnorm=True,
                    activate='LeakyReLU',
                    out_activate='Sigmoid',
                ),
                autoencoder=dict(
                    lr=1e-3,
                    hidden_dim=hidden_dim,
                    encoder=dict(
                        encs=[[784, 500],
                              [256, 500], ],

                        batchnorm=False,
                        activate='ReLU',
                        out_activate='ReLU',
                    ),
                    decoder=dict(
                        decs=[[500, 784],
                              [500, 256]],
                        batchnorm=False,
                        activate='ReLU',
                        out_activate=None,
                    ),
                )

            )
        ),
    )

    args.config = config
