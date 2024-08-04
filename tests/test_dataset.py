# # TODO: test

# def dummy_lgb_data_func(X: XdataLike, y: Optional[YdataLike] = None) -> Union[DataLike, XdataLike]:
#     return lgb.Dataset(data=X, label=y) if y is not None else X


# def dummy_xgb_data_func(X: XdataLike, y: Optional[YdataLike] = None) -> DataLike:
#     return xgb.DMatrix(data=X, label=y)


# df_data = pd.DataFrame([[1, 2], [3, 4]])
# series_data = pd.Series([1, 2])
# array_data = np.array([[1, 2], [3, 4]])
# array_label = np.array([1, 2])
# series_label = pd.Series([1,2])

# @pytest.mark.parametrize("func, input_data, label_data, expected_type", [
#     (dummy_lgb_data_func, df_data, None, XdataLike),
#     (dummy_lgb_data_func, series_data, None, XdataLike),
#     (dummy_lgb_data_func, df_data, series_label, DataLike),
#     (dummy_lgb_data_func, array_data, array_label, DataLike),
#     (dummy_xgb_data_func, df_data, None, DataLike),
#     (dummy_xgb_data_func, series_data, None, DataLike),
#     (dummy_xgb_data_func, df_data, series_label, DataLike),
#     (dummy_xgb_data_func, array_data, array_label,DataLike),
# ])
# def test_datafunclike(func, input_data, label_data, expected_type):
#     result = func(input_data, label_data)
#     assert isinstance(result, expected_type)
