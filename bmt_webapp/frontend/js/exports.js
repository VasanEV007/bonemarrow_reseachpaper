/**
 * BMT SURVIVAL PREDICTION — CSV & REPORT EXPORT MODULE
 * Handles safe CSV generation, formula-injection sanitization, and downloads.
 */

window.BMTExports = (function () {
  
  // Safe CSV escaping to prevent CSV formula injection (=, +, -, @)
  function sanitizeCell(val) {
    if (val === null || val === undefined) return '""';
    let s = String(val);
    if (/^[=+\-@\t\r]/.test(s)) {
      s = "'" + s; // Prefix with single quote to disable spreadsheet execution
    }
    // Escape double quotes
    s = s.replace(/"/g, '""');
    return `"${s}"`;
  }

  function downloadCsvBlob(content, filename) {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  // Export the active prediction record
  function exportPredictionRecord(predictionResult, patientInputs) {
    if (!predictionResult) {
      alert('No prediction has been generated yet. Please run inference first.');
      return;
    }

    const headers = [
      'prediction_id',
      'timestamp',
      'model_id',
      'model_name',
      'target_name',
      'predicted_value',
      'target_units',
      'probability_or_ci',
      'interpretation'
    ];

    const isClassif = predictionResult.prediction_class !== undefined;
    const predVal = isClassif ? predictionResult.prediction_class : predictionResult.prediction;
    const probOrCi = isClassif
      ? `${(predictionResult.positive_probability * 100).toFixed(1)}%`
      : `${predictionResult.lower_bound?.toFixed(1)} - ${predictionResult.upper_bound?.toFixed(1)} d`;

    const row = [
      sanitizeCell(predictionResult.prediction_id || `PRED-${Date.now()}`),
      sanitizeCell(new Date().toISOString()),
      sanitizeCell(predictionResult.model_id),
      sanitizeCell(predictionResult.model_name),
      sanitizeCell(predictionResult.target_name),
      sanitizeCell(predVal),
      sanitizeCell(predictionResult.target_units || 'N/A'),
      sanitizeCell(probOrCi),
      sanitizeCell(predictionResult.interpretation || '')
    ];

    // Append all 35 patient inputs as additional columns
    const inputKeys = Object.keys(patientInputs || {}).sort();
    inputKeys.forEach(k => {
      headers.push(`input_${k}`);
      row.push(sanitizeCell(patientInputs[k]));
    });

    const csvContent = headers.join(',') + '\r\n' + row.join(',') + '\r\n';
    downloadCsvBlob(csvContent, `patient_prediction_${Date.now()}.csv`);
  }

  // Export current 35 patient input values
  function exportPatientInputs(patientInputs) {
    if (!patientInputs || Object.keys(patientInputs).length === 0) {
      alert('Patient form is empty.');
      return;
    }

    const headers = ['feature_name', 'submitted_value'];
    const rows = Object.keys(patientInputs).sort().map(k => {
      return `${sanitizeCell(k)},${sanitizeCell(patientInputs[k])}`;
    });

    const csvContent = headers.join(',') + '\r\n' + rows.join('\r\n') + '\r\n';
    downloadCsvBlob(csvContent, `patient_inputs_${Date.now()}.csv`);
  }

  return {
    sanitizeCell,
    downloadCsvBlob,
    exportPredictionRecord,
    exportPatientInputs
  };
})();
