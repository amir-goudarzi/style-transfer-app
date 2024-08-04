%%bash
STYLE_PATH="style-transfer-app/examples/style1.jpg"
CONTENT_PATH="style-transfer-app/examples/new-york-city.jpg"
OUTPUT_PATH="./output_directory"
ALPHABETA_RATIO=1e4

python3 style-transfer-app/src/main.py \
    --style_path "$STYLE_PATH" \
    --content_path "$CONTENT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --alpha_beta_ratio "$ALPHABETA_RATIO"