# @app.command()
# def predict():
#     logger.remove()
#     logger.add(stderr, level="INFO")

#     nikkei225 = fetcher.load_nikkei225_csv()

#     model_bearish = CNNModel()
#     model_bearish.load_state_dict(torch.load("./model/Bearish_0.813.pth"))

#     model_neutral = CNNModel()
#     model_neutral.load_state_dict(torch.load("./model/Neutral_0.864.pth"))

#     model_bullish = CNNModel()
#     model_bullish.load_state_dict(torch.load("./model/Bullish_0.767.pth"))

#     i = 1
#     for code in nikkei225.get_column("code"):
#         df = fetcher.fetch_daily_quotes(code)
#         if df is None:
#             continue

#         stock_sliced = df.tail(10)
#         df_sampled = stock_sliced.with_columns(
#             pl.col("date").str.to_date().alias("date")
#         )
#         img = create_candlestick_chart(df_sampled, code, "predict")

#         image = Image.open(io.BytesIO(img)).convert("RGB")
#         image = torchvision.transforms.ToTensor()(image)
#         image = image.unsqueeze(0)

#         model_bullish.eval()
#         with torch.no_grad():
#             output = model_bullish(image)
#             predicted = torch.argmax(output, dim=1)
#             probability = torch.max(output, dim=1)

#             logger.info(f"{code} Bullish: {predicted} [確率: {probability.values}]")

#         model_neutral.eval()
#         with torch.no_grad():
#             output = model_neutral(image)
#             predicted = torch.argmax(output, dim=1)
#             probability = torch.max(output, dim=1)

#             logger.info(f"{code} Neutral: {predicted} [確率: {probability.values}]")

#         model_bearish.eval()
#         with torch.no_grad():
#             output = model_bearish(image)
#             predicted = torch.argmax(output, dim=1)
#             probability = torch.max(output, dim=1)

#             logger.info(f"{code} Bearish: {predicted} [確率: {probability.values}]")

#         if i % 10 == 0:
#             logger.info(f"{i}/225 has been processed.")
#         i += 1
