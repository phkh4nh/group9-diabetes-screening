/** Group 9 — Student-only Google Sheets logging via Google Apps Script.
 * Open the target Google Sheet > Extensions > Apps Script, paste this file.
 * Script property: INGEST_TOKEN = long secret, identical to Streamlit secrets.
 * Deploy: Web app, Execute as Me, Access Anyone (for Python HTTPS requests).
 * Anyone with the shared secret could submit rows. Use DEMO / SYNTHETIC data only.
 */
const SHEET_NAME = 'Predictions';
const EXPECTED_HEADERS = ["prediction_id", "timestamp", "model_version", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income", "predicted_probability", "risk_percentile", "top_feature_1", "top_feature_1_shap", "top_feature_2", "top_feature_2_shap", "top_feature_3", "top_feature_3_shap", "actual_diabetes_outcome", "outcome_recorded_at"];

function jsonResponse_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

function doGet() {
  return jsonResponse_({ok: true, service: 'group9-demo-log', method: 'POST required'});
}

function doPost(e) {
  let lock = null;
  try {
    const expectedToken = PropertiesService.getScriptProperties().getProperty('INGEST_TOKEN');
    if (!expectedToken || expectedToken.length < 24) throw new Error('INGEST_TOKEN not configured');
    if (!e || !e.postData || !e.postData.contents) throw new Error('Empty POST body');
    const data = JSON.parse(e.postData.contents);
    if (typeof data.token !== 'string' || data.token !== expectedToken) throw new Error('Unauthorized');
    const row = data.row;
    if (!Array.isArray(row) || row.length !== EXPECTED_HEADERS.length) throw new Error('Expected exactly 34 columns');
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(String(row[0])))
      throw new Error('Invalid prediction ID');
    if (!/^\d{4}-\d{2}-\d{2}T/.test(String(row[1]))) throw new Error('Invalid UTC timestamp');
    if (!/^[0-9A-Za-z._-]{1,40}$/.test(String(row[2]))) throw new Error('Invalid model version');
    // All 21 inputs are numeric; SHAP contributions are numeric. End two outcome fields must be empty.
    for (let i = 3; i <= 25; i++) {
      if (typeof row[i] !== 'number' || !Number.isFinite(row[i])) throw new Error('Invalid numeric input/score');
    }
    if (row[24] < 0 || row[24] > 1 || row[25] < 0 || row[25] > 100)
      throw new Error('Probability/percentile outside valid range');
    const features = EXPECTED_HEADERS.slice(3, 24);
    for (const i of [26, 28, 30]) {
      if (!features.includes(row[i]) || typeof row[i+1] !== 'number' || !Number.isFinite(row[i+1]))
        throw new Error('Invalid SHAP pair');
    }
    if (row[32] !== '' || row[33] !== '') throw new Error('Actual outcome must initially be blank');
    const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    if (!spreadsheet) throw new Error('Script must be created from the Google Sheet (Extensions > Apps Script)');
    const sheet = spreadsheet.getSheetByName(SHEET_NAME);
    if (!sheet) throw new Error('Sheet tab "Predictions" not found');
    const headers = sheet.getRange(1, 1, 1, EXPECTED_HEADERS.length).getValues()[0];
    if (JSON.stringify(headers) !== JSON.stringify(EXPECTED_HEADERS))
      throw new Error('Sheet header differs from the 34-column schema');
    lock = LockService.getScriptLock();
    lock.waitLock(15000);
    // Idempotency protects against accidental repeated POST with the same prediction_id.
    const last = sheet.getLastRow();
    if (last > 1) {
      const found = sheet.getRange(2, 1, last-1, 1)
        .createTextFinder(String(row[0])).matchEntireCell(true).findNext();
      if (found) return jsonResponse_({ok: true, prediction_id: String(row[0]), duplicate: true});
    }
    sheet.appendRow(row);
    SpreadsheetApp.flush();
    return jsonResponse_({ok: true, prediction_id: String(row[0])});
  } catch (err) {
    return jsonResponse_({ok: false, error: String(err && err.message || err)});
  } finally {
    if (lock) { try { lock.releaseLock(); } catch (ignored) {} }
  }
}
