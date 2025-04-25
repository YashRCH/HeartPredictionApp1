package com.example.heartpredictionapp;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;

public class MainActivity extends AppCompatActivity {

    private static final float[] MEAN = {54.3f, 150.0f, 0.6f, 1.2f};
    private static final float[] SCALE = {9.2f, 22.5f, 0.49f, 0.8f};
    private static final float THRESHOLD = 0.5f;
    private static final String TAG = "HeartPredictionApp";

    private OrtEnvironment env;
    private OrtSession session;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        EditText inputAge = findViewById(R.id.inputAge);
        EditText inputThalach = findViewById(R.id.inputThalach);
        RadioGroup sexGroup = findViewById(R.id.sexGroup);
        Spinner cpSpinner = findViewById(R.id.cpSpinner);
        Button predictButton = findViewById(R.id.predictButton);
        TextView resultText = findViewById(R.id.resultText);

        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(
                this,
                R.array.chest_pain_types,
                android.R.layout.simple_spinner_item
        );
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        cpSpinner.setAdapter(adapter);

        new Thread(this::loadModel).start();

        predictButton.setOnClickListener(v -> {
            if (validateInputs(inputAge, inputThalach)) {
                new Thread(() -> {
                    try {
                        float[] features = getFeatures(inputAge, inputThalach, sexGroup, cpSpinner);
                        float rawScore = predict(features);
                        runOnUiThread(() -> displayResult(resultText, rawScore));
                    } catch (Exception e) {
                        runOnUiThread(() -> handlePredictionError(e));
                    }
                }).start();
            }
        });
    }

    private void loadModel() {
        try {
            env = OrtEnvironment.getEnvironment();
            try (InputStream is = getAssets().open("xgb_heart_model.onnx")) {
                byte[] modelBytes = new byte[is.available()];
                int bytesRead = is.read(modelBytes);
                if (bytesRead != modelBytes.length) {
                    throw new RuntimeException("Failed to read complete model file");
                }
                session = env.createSession(modelBytes, new OrtSession.SessionOptions());
                Log.d(TAG, "Model loaded successfully");
            }
        } catch (Exception e) {
            Log.e(TAG, "Model loading failed", e);
            runOnUiThread(() -> Toast.makeText(this, "Failed to load model: " + e.getMessage(), Toast.LENGTH_LONG).show());
        }
    }

    private boolean validateInputs(EditText ageInput, EditText thalachInput) {
        try {
            if (ageInput.getText().toString().isEmpty() || thalachInput.getText().toString().isEmpty()) {
                Toast.makeText(this, "Please fill all fields", Toast.LENGTH_SHORT).show();
                return false;
            }

            float age = Float.parseFloat(ageInput.getText().toString());
            if (age < 20 || age > 100) {
                Toast.makeText(this, "Please enter a valid age (20–100)", Toast.LENGTH_SHORT).show();
                return false;
            }

            float thalach = Float.parseFloat(thalachInput.getText().toString());
            if (thalach < 60 || thalach > 220) {
                Toast.makeText(this, "Please enter a valid heart rate (60–220)", Toast.LENGTH_SHORT).show();
                return false;
            }

            return true;
        } catch (NumberFormatException e) {
            Toast.makeText(this, "Invalid number format", Toast.LENGTH_SHORT).show();
            return false;
        }
    }

    private float[] getFeatures(EditText ageInput, EditText thalachInput,
                                RadioGroup sexGroup, Spinner cpSpinner) {
        float age = Float.parseFloat(ageInput.getText().toString());
        float thalach = Float.parseFloat(thalachInput.getText().toString());
        int sexId = sexGroup.getCheckedRadioButtonId();
        float sex = (sexId == R.id.male) ? 1.0f : 0.0f;
        float cp = Float.parseFloat(cpSpinner.getSelectedItem().toString().split(" - ")[0]);

        float[] features = {age, thalach, sex, cp};
        for (int i = 0; i < features.length; i++) {
            features[i] = (features[i] - MEAN[i]) / SCALE[i];
        }
        Log.d(TAG, "Normalized features: " + java.util.Arrays.toString(features));
        return features;
    }

    private float predict(float[] features) throws Exception {
        if (session == null) {
            throw new IllegalStateException("Model not loaded");
        }

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(features),
                new long[]{1, features.length}
        )) {
            OrtSession.Result output = session.run(Collections.singletonMap("float_input", inputTensor));
            Object result = output.get(0).getValue();

            if (result instanceof float[][]) {
                return ((float[][]) result)[0][0];
            } else if (result instanceof long[][]) {
                return ((long[][]) result)[0][0];
            } else if (result instanceof float[]) {
                return ((float[]) result)[0];
            } else if (result instanceof long[]) {
                return ((long[]) result)[0];
            } else {
                throw new RuntimeException("Unexpected output type: " + result.getClass());
            }
        }
    }

    private void displayResult(TextView resultText, float score) {
        String result;
        int color;
        int bgColor;
        String recommendation;

        if (score > THRESHOLD) {
            result = "High Risk of Heart Disease (" + String.format("%.1f%%", score * 100) + ")";
            recommendation = "\nRECOMMENDATION: Please consult a physician.";
            color = ContextCompat.getColor(this, R.color.high_risk);
            bgColor = ContextCompat.getColor(this, R.color.high_risk_bg);
        } else {
            result = "Low Risk of Heart Disease (" + String.format("%.1f%%", (1 - score) * 100) + ")";
            recommendation = "\nRECOMMENDATION: No consultation needed at this time.";
            color = ContextCompat.getColor(this, R.color.low_risk);
            bgColor = ContextCompat.getColor(this, R.color.low_risk_bg);
        }

        resultText.setText(result + recommendation);
        resultText.setTextColor(color);
        resultText.setBackgroundColor(bgColor);
    }

    private void handlePredictionError(Exception e) {
        Log.e(TAG, "Prediction error", e);
        Toast.makeText(this, "Prediction error: " + e.getMessage(), Toast.LENGTH_LONG).show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
        } catch (Exception e) {
            Log.e(TAG, "Error cleaning up ONNX resources", e);
        }
    }
}
