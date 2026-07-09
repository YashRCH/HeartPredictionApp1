package com.example.heartpredictionapp;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.button.MaterialButtonToggleGroup;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.progressindicator.CircularProgressIndicator;
import com.google.android.material.textfield.TextInputEditText;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Locale;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {

    // StandardScaler parameters are loaded at runtime from assets/scaler_params.json
    // (generated from the trained scaler.pkl), in the model's feature order: cp, thalach, sex, age.
    // Full precision matters: XGBoost is a step function, so rounded constants can flip a
    // prediction near a tree-split boundary. Scaling is done in double, then cast to float
    // for the tensor, to match the Python training pipeline exactly.
    private double[] scalerMean, scalerScale;

    private static final float THRESHOLD = 0.5f;
    private static final String TAG = "HeartPredictionApp";

    private OrtEnvironment env;
    private OrtSession session;

    private TextInputEditText inputAge, inputThalach;
    private MaterialButtonToggleGroup genderToggle;
    private AutoCompleteTextView cpDropdown;
    private MaterialButton predictButton;
    private CircularProgressIndicator loadingIndicator;
    private MaterialCardView resultCard;
    private TextView riskPercent, riskLabel, riskRecommendation;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        setContentView(R.layout.activity_main);

        inputAge = findViewById(R.id.inputAge);
        inputThalach = findViewById(R.id.inputThalach);
        genderToggle = findViewById(R.id.genderToggle);
        cpDropdown = findViewById(R.id.cpDropdown);
        predictButton = findViewById(R.id.predictButton);
        loadingIndicator = findViewById(R.id.loadingIndicator);
        resultCard = findViewById(R.id.resultCard);
        riskPercent = findViewById(R.id.riskPercent);
        riskLabel = findViewById(R.id.riskLabel);
        riskRecommendation = findViewById(R.id.riskRecommendation);

        applyEdgeToEdgeInsets();
        setupChestPainDropdown();

        // Disable prediction until the model has finished loading off the main thread.
        predictButton.setEnabled(false);
        loadingIndicator.show();
        new Thread(this::loadModel).start();

        predictButton.setOnClickListener(v -> {
            if (!validateInputs()) return;
            loadingIndicator.show();
            predictButton.setEnabled(false);
            new Thread(() -> {
                try {
                    float[] features = buildScaledFeatures();
                    float risk = predict(features);
                    runOnUiThread(() -> {
                        loadingIndicator.hide();
                        predictButton.setEnabled(true);
                        displayResult(risk);
                    });
                } catch (Exception e) {
                    runOnUiThread(() -> {
                        loadingIndicator.hide();
                        predictButton.setEnabled(true);
                        handlePredictionError(e);
                    });
                }
            }).start();
        });
    }

    private void applyEdgeToEdgeInsets() {
        View root = findViewById(R.id.rootScroll);
        final int base = Math.round(24 * getResources().getDisplayMetrics().density);
        ViewCompat.setOnApplyWindowInsetsListener(root, (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(base, base + bars.top, base, base + bars.bottom);
            return insets;
        });
        ViewCompat.requestApplyInsets(root);
    }

    private void setupChestPainDropdown() {
        ArrayAdapter<CharSequence> adapter = new ArrayAdapter<>(
                this, R.layout.list_item_dropdown,
                getResources().getStringArray(R.array.chest_pain_types));
        cpDropdown.setAdapter(adapter);
        // Default selection so a value is always present without opening the menu.
        cpDropdown.setText(adapter.getItem(0), false);
    }

    private void loadModel() {
        try {
            loadScalerParams();
            env = OrtEnvironment.getEnvironment();
            try (InputStream is = getAssets().open("xgb_heart_model.onnx")) {
                byte[] modelBytes = readAll(is);
                session = env.createSession(modelBytes, new OrtSession.SessionOptions());
                Log.d(TAG, "Model loaded successfully (" + modelBytes.length + " bytes)");
            }
            runOnUiThread(() -> {
                loadingIndicator.hide();
                predictButton.setEnabled(true);
            });
        } catch (Exception e) {
            Log.e(TAG, "Model loading failed", e);
            runOnUiThread(() -> {
                loadingIndicator.hide();
                predictButton.setEnabled(false);
                Toast.makeText(this, R.string.error_model_load, Toast.LENGTH_LONG).show();
            });
        }
    }

    private static byte[] readAll(InputStream is) throws java.io.IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] chunk = new byte[8192];
        int n;
        while ((n = is.read(chunk)) != -1) {
            buffer.write(chunk, 0, n);
        }
        return buffer.toByteArray();
    }

    private void loadScalerParams() throws Exception {
        try (InputStream is = getAssets().open("scaler_params.json")) {
            JSONObject obj = new JSONObject(new String(readAll(is), StandardCharsets.UTF_8));
            scalerMean = toDoubleArray(obj.getJSONArray("mean"));
            scalerScale = toDoubleArray(obj.getJSONArray("scale"));
        }
    }

    private static double[] toDoubleArray(JSONArray arr) throws org.json.JSONException {
        double[] out = new double[arr.length()];
        for (int i = 0; i < out.length; i++) out[i] = arr.getDouble(i);
        return out;
    }

    private boolean validateInputs() {
        String ageStr = text(inputAge);
        String thalachStr = text(inputThalach);

        if (ageStr.isEmpty() || thalachStr.isEmpty()) {
            Toast.makeText(this, R.string.error_fill_fields, Toast.LENGTH_SHORT).show();
            return false;
        }
        try {
            float age = Float.parseFloat(ageStr);
            if (age < 20 || age > 100) {
                Toast.makeText(this, R.string.error_age_range, Toast.LENGTH_SHORT).show();
                return false;
            }
            float thalach = Float.parseFloat(thalachStr);
            if (thalach < 60 || thalach > 220) {
                Toast.makeText(this, R.string.error_hr_range, Toast.LENGTH_SHORT).show();
                return false;
            }
            return true;
        } catch (NumberFormatException e) {
            Toast.makeText(this, R.string.error_number_format, Toast.LENGTH_SHORT).show();
            return false;
        }
    }

    // Builds the standardized feature vector in the model's training order: cp, thalach, sex, age.
    private float[] buildScaledFeatures() {
        double age = Double.parseDouble(text(inputAge));
        double thalach = Double.parseDouble(text(inputThalach));
        double sex = (genderToggle.getCheckedButtonId() == R.id.btnMale) ? 1.0 : 0.0;
        double cp = parseLeadingInt(cpDropdown.getText().toString());

        double[] raw = {cp, thalach, sex, age};
        float[] features = new float[raw.length];
        for (int i = 0; i < raw.length; i++) {
            features[i] = (float) ((raw[i] - scalerMean[i]) / scalerScale[i]);
        }
        Log.d(TAG, "Scaled features [cp,thalach,sex,age]: " + java.util.Arrays.toString(features));
        return features;
    }

    private float predict(float[] features) throws Exception {
        if (session == null) {
            throw new IllegalStateException("Model not loaded");
        }
        try (OnnxTensor input = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(features), new long[]{1, features.length});
             OrtSession.Result output = session.run(
                     Collections.singletonMap("float_input", input))) {

            OnnxValue probs = output.get("probabilities")
                    .orElseThrow(() -> new IllegalStateException("Model has no 'probabilities' output"));
            Object value = probs.getValue();
            // probabilities is a float tensor [1, 2] -> [P(no disease), P(disease)].
            if (value instanceof float[][]) {
                return ((float[][]) value)[0][1];
            } else if (value instanceof float[]) {
                return ((float[]) value)[1];
            }
            throw new RuntimeException("Unexpected probabilities type: " + value.getClass());
        }
    }

    private void displayResult(float risk) {
        boolean high = risk > THRESHOLD;
        int accent = ContextCompat.getColor(this, high ? R.color.risk_high : R.color.risk_low);
        int container = ContextCompat.getColor(this, high ? R.color.risk_high_bg : R.color.risk_low_bg);

        resultCard.setCardBackgroundColor(container);
        resultCard.setStrokeColor(accent);

        riskPercent.setVisibility(View.VISIBLE);
        riskPercent.setText(String.format(Locale.US, "%.0f%%", risk * 100f));
        riskPercent.setTextColor(accent);

        riskLabel.setText(high ? R.string.risk_high_label : R.string.risk_low_label);
        riskLabel.setTextColor(accent);

        riskRecommendation.setVisibility(View.VISIBLE);
        riskRecommendation.setText(high ? R.string.risk_high_reco : R.string.risk_low_reco);
    }

    private void handlePredictionError(Exception e) {
        Log.e(TAG, "Prediction error", e);
        Toast.makeText(this, getString(R.string.error_prediction, e.getMessage()), Toast.LENGTH_LONG).show();
    }

    private String text(TextView view) {
        return view.getText() == null ? "" : view.getText().toString().trim();
    }

    // Parses the numeric code from an entry like "0 - Typical angina".
    private int parseLeadingInt(String entry) {
        String trimmed = entry.trim();
        int i = 0;
        while (i < trimmed.length() && Character.isDigit(trimmed.charAt(i))) i++;
        return i == 0 ? 0 : Integer.parseInt(trimmed.substring(0, i));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            if (session != null) session.close();
            if (env != null) env.close();
        } catch (Exception e) {
            Log.e(TAG, "Error cleaning up ONNX resources", e);
        }
    }
}
