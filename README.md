### KWS-project

To run streaming:

```bash
pip install -r requirements
python stream.py --path="checkpoints/final_model_streaming.pt"
```

You can find best model binary in`checkpoints/final_model_streaming.pt`. Note that model includes melspec transform, so it takes raw audio chunk as an input.
