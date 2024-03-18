"""
for each original object
    calculate num. local minima

    if number local minima > 1 and px count > 900 px
        for each height layer
            label layer & count num. sub_objects
            calculate ||num. sub_objects - num. local minima||

        choose layer with min ||...||
        if same ||...|| for multiple layers
            choose highest one

    remove information of original object above chosen layer
    """


