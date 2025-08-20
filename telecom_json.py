import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import re
import json
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=api_key,
    model_name="llama3-8b-8192",
    temperature=0.0,
    top_p=1.0,
    max_tokens=2048,
)

prompt_template = PromptTemplate(
    input_variables=["ocr"],
    template = """
You are an expert at reading and extracting structured data from invoice documents. And avoid Preamble and any other information, ONLY JSON

Given the following raw invoice text, extract the structured JSON in the following format:

{{
  "invoiceSummary": {{
    "vendorName": "...",
    "invoiceNumber": "...",
    "accountNumber": "...",
    "billDate": "DD-MM-YYYY",
    "dueDate": "DD-MM-YYYY",
    "previousBalance": ...,
    "payments": ...,
    "pastDue": ...,
    "adjustments": ...,
    "currentCharges": ...,
    "amountDue": ...,
    "serviceToName": "...",
    "serviceToAddress1": "...",
    "serviceToAddress2": "...",
    "serviceToCity": "...",
    "serviceToState": "...",
    "serviceToZip": "...",
    "remitToName": "...",
    "remitToAddress1": "...",
    "remitToAddress2": "...",
    "remitToCity": "...",
    "remitToState": "...",
    "remitToZip": "...",
    "billToName": "...",
    "billToAddress": "...",
    "billToAddress2": "...",
    "billToCity": "...",
    "billToState": "...",
    "billToZip": "..."
  }},
  "InvoiceData": {{
    "Section Title 1": [[
      {{
        "description": "...",
        "amount": ...,
        "children": [[
          {{
            "description": "...",
            "amount": ...
          }}
        ]]
      }}
    ]],
    "Section Title 2": [[
      {{
        "description": "...",
        "amount": ...
      }}
    ]]
  }}
}}

## Rules:

### For `invoiceSummary`:
- Extract only the listed fields.
- If a value is missing in the document, use `null`.
- Dates must be in "YYYY-MM-DD" format.
- All monetary values must be numbers (no currency symbols or commas).

### For `InvoiceDetails`:
- Section titles must be inferred from the document headers (e.g., "Monthly Charges", "Taxes").
- Each section must contain an array of objects with at least:
  - `description`: The item name or service
  - `amount`: The associated cost as a number
- If a line item contains sub-items (e.g., bundled services), include them under a `children` array with the same structure.

### Final Output:
- Return a single valid JSON object combining both `invoiceSummary` and `detailedBreakdown`.
- Do not include any explanatory text or comments — only the raw JSON object.


INVOICEINFO = {{
    "type": "function",
    "name": "extract_invoice_data",
    "description": "Extract structured data from an invoice document text",
    "parameters": {{
        "type": "object",
        "properties": {{
            "invoiceSummary": {{
            "type": "object",
            "description": "Fixed summary fields extracted from the invoice header",
            "properties": {{
                "vendorName": {{
                    "type": "string",
                    "description": "Topmost vendor name/logo on the invoice, usually at the top-left or center. Also used as 'remitToName'."
                }},
                "invoiceNumber": {{
                    "type": "string",
                    "description": "Extract from labels like 'Invoice Number', 'Invoice', 'Bill Number', etc. If missing, construct using 'accountNumber_YYMM'."
                }},
                "accountNumber": {{
                    "type": "string",
                    "description": "Extract from labels like 'Account Number', 'Invoice Account', 'Customer Number'. Remove symbols like '-' or '#' and return only digits."
                }},
                "billDate": {{
                    "type": "string",
                    "description": "Use 'Invoice Date', 'Statement Date', or 'Read Date'. If 'Billing Period: XX – YY' is present, use end date (YY). Format: DD-MM-YYYY."
                    }},
                "dueDate": {{
                    "type": "string",
                    "description": "Extract from 'Due Date', 'Required Payment Date', etc. If missing, add 20 days to 'billDate'. Format: DD-MM-YYYY."
                }},
                "previousBalance": {{
                    "type": "number",
                    "description": "Use values under 'Previous Balance', 'Balance Last Statement', or 'Amount of Last Bill'."
                }},
                "payments": {{
                    "type": "number",
                    "description": "Amount already paid. Extract from 'Payments', 'Credits Applied', or 'Payments Received'. If payment exceeds previous balance, subtract excess from 'currentCharges'."
                }},
                "pastDue": {{
                    "type": "number",
                    "description": "Amount unpaid from previous cycle. Usually calculated as previousBalance - payments. If explicitly mentioned, use that value."
                }},
                "adjustments": {{
                    "type": "number",
                    "description": "Amount of adjustments (credits/debits) if present; otherwise, use 0."
                }},
                "currentCharges": {{
                    "type": "number",
                    "description": "Extract from 'Current Charges', 'New Charges', or 'Monthly Charges'."
                }},
                "amountDue": {{
                    "type": "number",
                    "description": "Must equal: adjustments + pastDue + currentCharges. Extract if stated, or compute."
                }},
                "serviceToName": {{
                    "type": "string",
                    "description": "Same as 'vendorName'."
                }},
                "serviceToAddress1": {{
                    "type": "string",
                    "description": "Which is different from the remit address, where service is provided "
                }},
                "remitToAddress2": {{
                    "type": "string",
                    "description": "Second line of service address if available."
                }},
                "remitToCity": {{
                    "type": "string",
                    "description": "City from 'service To' address."
                }},
                "remitToState": {{
                    "type": "string",
                    "description": "Two-letter U.S. state code for service address."
                }},
                "remitToZip": {{
                    "type": "string",
                    "description": "ZIP code from 'service To' address."
                }},
                "remitToName": {{
                    "type": "string",
                    "description": "Same as 'vendorName'."
                }},
                "remitToAddress1": {{
                    "type": "string",
                    "description": "First line under 'Remit To' address or right-side vendor address."
                }},
                "remitToAddress2": {{
                    "type": "string",
                    "description": "Second line of remit address if available."
                }},
                "remitToCity": {{
                    "type": "string",
                    "description": "City from 'Remit To' address."
                }},
                "remitToState": {{
                "type": "string",
                "description": "Two-letter U.S. state code for remit address. Must match predefined client list (State Codes.xlsx)."
                }},
                "remitToZip": {{
                "type": "string",
                "description": "ZIP code from 'Remit To' address."
                }},
                "billToName": {{
                "type": "string",
                "description": "Client name under 'Bill To'. If not available in the invoice, refer external source using account number."
                }},
                "billToAddress": {{
                "type": "string",
                "description": "First line under 'Bill To' address."
                }},
                "billToAddress2": {{
                "type": "string",
                "description": "Second line under 'Bill To' address (optional)."
                }},
                "billToCity": {{
                "type": "string",
                "description": "City from 'Bill To' address."
                }},
                "billToState": {{
                "type": "string",
                "description": "Two-letter U.S. state code from 'Bill To' address."
                }},
                "billToZip": {{
                "type": "string",
                "description": "ZIP code from 'Bill To' address."
                }}
            }},
            "required": ["vendorName", "invoiceNumber", "amountDue"]
            }}
            "InvoiceDetails": {{
                "type": "object",
                "description": "Flexible breakdown of charges, taxes, or other grouped sections, dynamically inferred from invoice",
                "patternProperties": {{
                    ".*": {{
                        "type": "array",
                        "description": "Array of line items for each inferred section (e.g., Charges, Services, Taxes)",
                        "items": {{
                            "type": "object",
                            "properties": {{
                                "description": {{
                                    "type": "string",
                                    "description": "Text description of the line item"
                                }},
                                "amount": {{
                                    "type": "number",
                                    "description": "Monetary value of the line item"
                                }},
                                "children": {{
                                    "type": "array",
                                    "description": "Sub-items under this line item (if applicable)",
                                    "items": {{
                                        "type": "object",
                                        "properties": {{
                                            "description": {{
                                                "type": "string",
                                                "description": "Text description of the child line item"
                                            }},
                                            "amount": {{
                                                "type": "number",
                                                "description": "Monetary value of the child item"
                                            }}
                                        }},
                                        "required": ["description", "amount"]
                                    }}
                                }}
                            }},
                            "required": ["description", "amount"]
                        }}
                    }}
                }}
            }}
        }},
        "required": ["invoiceSummary", "InvoiceDetails"]
    }}
}}

{ocr}

"""
)

chain = LLMChain(llm=llm, prompt=prompt_template)

def extract_invoice_json_telecom(ocr_text):
    """
    Given invoice OCR text, return structured JSON as a dict.
    Handles model formatting issues and safely parses JSON.
    """
    raw_response = chain.run(ocr=ocr_text)

    raw_response = raw_response.strip()
    match = re.search(r"```(?:json)?(.*?)```", raw_response, re.DOTALL)
    if match:
        raw_response = match.group(1).strip()
    raw_response = re.sub(r'^(Here is the extracted JSON.*?:)', '', raw_response, flags=re.IGNORECASE).strip()
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as e:
        return {"error": "Failed to decode JSON from model output", "details": str(e), "raw": raw_response}
