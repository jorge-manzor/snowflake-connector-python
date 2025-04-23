# Google Sheet Automatic Cell Modified Date

1. Go to extensions > Apps Script
2. Paste the following code:

```gs
    function onEdit(e) {
    // Get information about what was edited
    var range = e.range;
    var sheet = range.getSheet();
    var sheetName = sheet.getName();
    var row = range.getRow();
    var col = range.getColumn();
    
    // Only run on the sheet you want to monitor
    // Replace "Sheet1" with your sheet name if needed
    if (sheetName == "OWA - Manual Reconciliation") {
        
        // Check if the edit was in columns A, B, C or D (columns 1, 2, 3 or 4)
        if (col >= 1 && col <= 4) {
        
        // Update the timestamp in column D (column 4) of the same row
        sheet.getRange(row, 5).setValue(new Date());
        
        // Optional: Format the timestamp cell
        sheet.getRange(row, 5).setNumberFormat("yyyy-mm-dd HH:mm:ss.ssssss");
        }
    }
    }
```
3. Specify the columns that will be monitored to trigger the tiestamp, and specify which column will display the timestamp.