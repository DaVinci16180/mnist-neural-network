from neural_network import get_formatted_img_from_path, gradient_descent, make_predictions, test_prediction, translate_output

W1, b1, W2, b2 = gradient_descent(0.10, 1000)

x = get_formatted_img_from_path('1.png')
saida = make_predictions(x, W1, b1, W2, b2)

print(saida)
translate_output(saida)

# Davi: 6
# William: 4