from neural_network import get_formatted_img_from_path, gradient_descent, make_predictions, test_prediction

W1, b1, W2, b2 = gradient_descent(0.10, 1000)

x = get_formatted_img_from_path('')

print(make_predictions(x, W1, b1, W2, b2))