def my_model(args):
    hidden_dim = args.hidden_dim
    config = dict(

        epochs=150,
        pre_epochs=200,
        views_select=dict(
            voc=[0, 1],
            mnist=[0, 1],
            Caltech=[3, 4],
        ),
        needpretrain=True,

        network=dict(
            voc=dict(
                standard_method='L2',  # MinMax, L2, Standard
                batch_size=9999999,
                seed=5,  # 5: 63
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
                standard_method='MinMax',  # MinMax, L2, Standard
                batch_size=9999999,
                seed=1,  # 0:88
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
                        encs=[[784, 1000, 1000, 1000],
                              [256, 1000, 1000, 1000], ],

                        batchnorm=True,
                        activate='ReLU',
                        out_activate='Softmax',
                    ),
                    decoder=dict(
                        decs=[[1000, 1000, 1000, 784],
                              [1000, 1000, 1000, 256]],
                        batchnorm=True,
                        activate='ReLU',
                        out_activate=None,
                    ),
                )

            ),
            Caltech=dict(
                standard_method='MinMax',  # MinMax, L2, Standard
                batch_size=9999999,
                seed=0,  # 3:88
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
                        encs=[[48, 1024, 1024],
                              [40, 1024, 1024],
                              [254, 1024, 1024],
                              [1984, 1024, 1024, 1024],
                              [512, 1024, 1024, 1024],
                              [928, 1500, 1024],
                              ],

                        batchnorm=True,
                        activate='ReLU',
                        out_activate='Softmax',
                    ),
                    decoder=dict(
                        decs=[[1024, 1024, 48],
                              [1024, 1024, 40],
                              [1024, 1024, 254],
                              [1024, 1024, 1024, 1984],
                              [1024, 1024, 1024, 512],
                              [1500, 1024, 928], ],
                        batchnorm=True,
                        activate='ReLU',
                        out_activate='ReLU',
                    ),
                )

            )
        ),
    )

    args.config = config


def Completer(args):
    hidden_dim = args.hidden_dim
    config = dict(
        epochs=150,
        pre_epochs=100,
        views_select=dict(
            voc=[0, 1],
            mnist=[0, 1],
            Caltech=[3, 4],
        ),
        needpretrain=True,

        network=dict(
            voc=dict(
                standard_method='L2',  # MinMax, L2, Standard
                batch_size=9999999,
                seed=5,  # 5: 63
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
                standard_method='MinMax',  # MinMax, L2, Standard
                batch_size=9999999,
                seed=1,  # 0:88
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
                    lr=1e-4,
                    hidden_dim=hidden_dim,
                    encoder=dict(
                        encs=[[784, 1024, 1024, 1024],
                              [256, 1024, 1024, 1024], ],

                        batchnorm=True,
                        activate='ReLU',
                        out_activate='Softmax',
                    ),
                    decoder=dict(
                        decs=[[1024, 1024, 1024, 784],
                              [1024, 1024, 1024, 256]],
                        batchnorm=True,
                        activate='ReLU',
                        out_activate='ReLU',
                    ),
                )

            ),
            Caltech=dict(
                standard_method='MinMax',  # MinMax, L2, Standard
                batch_size=9999999,
                seed=0,  # 3:88
                n_classes=20,

                autoencoder=dict(
                    lr=1e-4,
                    hidden_dim=hidden_dim,
                    encoder=dict(
                        encs=[[48, 1024],
                              [40, 1024],
                              [254, 1024],
                              [1984, 1024, 1024, 1024],
                              [512, 1024, 1024, 1024],
                              [928, 1500],
                              ],

                        batchnorm=True,
                        activate='ReLU',
                        out_activate='Softmax',
                    ),
                    decoder=dict(
                        decs=[[1024, 48],
                              [1024, 40],
                              [1024, 254],
                              [1024, 1024, 1024, 1984],
                              [1024, 1024, 1024, 512],
                              [1500, 928], ],
                        batchnorm=True,
                        activate='ReLU',
                        out_activate='ReLU',
                    ),
                )

            )
        ),
    )

    args.config = config
