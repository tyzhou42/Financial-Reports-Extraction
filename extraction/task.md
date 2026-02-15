## Task

- Extract all semiconductor reports.
- If older reports are difficult to process, prioritize reports in the new format and add more reports from a relevant sector (e.g. tech)

#### Required Modifications

- Add support for a low-cost LLM API (e.g., DeepSeek).
- Extract all analyst names and store them as separate columns: `analyst_1`, `analyst_2`, … (use `NaN` if none).
- Evaluate a stronger OCR package for improved table extraction.
- Enhance the VLM fallback module.
- Extract as many relevant metadata fields as possible.

#### Output Format

- CSV file

**Columns:**
`index`, `file_name`, `report_text`, `report_text_without_template`, `metadata_feature[1:m]`, `label(numerical, both S6 and PMAFE)`

