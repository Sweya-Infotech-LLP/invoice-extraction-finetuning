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
    template = '''

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
  }},
  "FeatureChargeDetails": [[
    {{
      "description1": "...",
      "description2": "...",
      "type": "...",
      "meterNumber": "...",
      "meterType": "...",
      "serviceName": "...",
      "serviceAddress": "...",
      "amount": ...,
      "discountAmount": ...,
      "unitQuantity": ...
    }}
  ]],
  "InvoiceAdjustments": {{
    "adjustmentAmount": ...,
    "adjustmentDesc": "..."
  }}
}}


## Rules:

### For `invoiceSummary`:
- Extract only the listed fields.
- If a value is missing in the document, use `null`.
- Dates must be in "YYYY-MM-DD" format.
- All monetary values must be numbers (no currency symbols or commas).

### For `InvoiceData`:
- Section titles must be inferred from the document headers (e.g., "Monthly Charges", "Taxes").
- Each section must contain an array of objects with at least:
  - `description`: The item name or service
  - `amount`: The associated cost as a number
- If a line item contains sub-items (e.g., bundled services), include them under a `children` array with the same structure.

### For `FeatureChargeDetails`:
- Each item must extract:
  - `description1`, `description2`, `type`, `meterNumber`, `meterType`, `serviceName`, `serviceAddress`, `amount`, `discountAmount`, `unitQuantity`.

### For `InvoiceAdjustments`:
- Only return the `adjustmentAmount` and `adjustmentDesc`.

### Final Output:
- Return a single valid JSON object combining all four sections above.
- Do not include any explanatory text or comments — only the raw JSON object.

INVOICE_INFO = {{
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
            "description": "Name of the vendor mentioned on the invoice"
          }},
          "invoiceNumber": {{
            "type": "string",
            "description": "Unique invoice number or reference ID"
          }},
          "accountNumber": {{
            "type": "string",
            "description": "Customer's or vendor's account number (if mentioned) remove any unwanted symbols like(#, -) return only numbers."
          }},
          "billDate": {{
            "type": "string",
            "description": "Date the invoice was issued (format: YYYY-MM-DD)"
          }},
          "dueDate": {{
            "type": "string",
            "description": "Date by which the payment is due (format: YYYY-MM-DD)"
          }},
          "previousBalance": {{
            "type": "number",
            "description": "Balance amount carried forward from the previous bill"
          }},
          "payments": {{
            "type": "number",
            "description": "Payments already made toward the balance"
          }},
          "pastDue": {{
            "type": "number",
            "description": "Past due amount remaining unpaid"
          }},
          "adjustments": {{
            "type": "number",
            "description": "Any adjustments made to the bill (credits/debits)"
          }},
          "currentCharges": {{
            "type": "number",
            "description": "Charges incurred in the current billing cycle"
          }},
          "amountDue": {{
            "type": "number",
            "description": "Total amount due on the invoice"
          }},
          "remitToName": {{
            "type": "string",
            "description": "Always same as VendorName"
          }},
          "remitToAddress1": {{
            "type": "string",
            "description": "Primary address line for remittance"
          }},
          "remitToAddress2": {{
            "type": "string",
            "description": "Secondary address line for remittance (optional)"
          }},
          "remitToCity": {{
            "type": "string",
            "description": "City for the remittance address"
          }},
          "remitToState": {{
            "type": "string",
            "description": "State for the remittance address"
          }},
          "remitToZip": {{
            "type": "string",
            "description": "ZIP or postal code for the remittance address"
          }},
          "billToName": {{
            "type": "string",
            "description": "Name of the individual or entity being billed"
          }},
          "billToAddress": {{
            "type": "string",
            "description": "Billing address of the recipient"
          }},
          "billToAddress2": {{
            "type": "string",
            "description": "Secondary address line for of the recipient(Optional)"
          }},
          "billToCity": {{
            "type": "string",
            "description": "City for the billing address"
          }},
          "billToState": {{
            "type": "string",
            "description": "State for the billing address"
          }},
          "billToZip": {{
            "type": "string",
            "description": "ZIP or postal code for the billing address"
          }}
        }},
        "required": ["vendorName", "invoiceNumber", "amountDue"]
      }},
      "InvoiceDetails": {{
        "type": "object",
        "description": "Flexible breakdown of charges, taxes, or other grouped sections, dynamically inferred from invoice. Keys are section headers like 'Monthly Charges', 'Taxes', etc. Each section is an array of line item objects with optional children.",
        "patternProperties": {{
          ".*": {{
            "type": "array",
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
      }},
      "FeatureChargeDetails": {{
        "type": "array",
        "description": "Detailed list of feature-level charges such as meters, services, or line items that are billed individually. Each object contains metering and service-related charge metadata.",
        "items": {{
          "type": "object",
          "properties": {{
            "description1": {{
              "type": "string",
              "description": "Primary description of the charge"
            }},
            "description2": {{
              "type": "string",
              "description": "Secondary description of the charge"
            }},
            "type": {{
              "type": "string",
              "description": "Type/category of the charge"
            }},
            "meterNumber": {{
              "type": "string",
              "description": "Meter number if applicable"
            }},
            "meterType": {{
              "type": "string",
              "description": "Type of the meter"
            }},
            "serviceName": {{
              "type": "string",
              "description": "Name of the service being billed"
            }},
            "serviceAddress": {{
              "type": "string",
              "description": "Address where the service was provided"
            }},
            "amount": {{
              "type": "number",
              "description": "Total charge amount for the feature"
            }},
            "discountAmount": {{
              "type": "number",
              "description": "Discount applied to this feature charge"
            }},
            "unitQuantity": {{
              "type": "number",
              "description": "Quantity of units used or billed"
            }}
          }}
        }}
      }},
      "InvoiceAdjustments": {{
        "type": "object",
        "description": "Any invoice-level adjustment entries such as corrections or manual overrides to the total bill",
        "properties": {{
          "adjustmentAmount": {{
            "type": "number",
            "description": "Amount of adjustment (positive or negative)"
          }},
          "adjustmentDescription": {{
            "type": "string",
            "description": "Text description of what the adjustment is for"
          }}
        }}
      }}
    }},
    "required": ["invoiceSummary", "InvoiceDetails", "FeatureChargeDetails", "InvoiceAdjustments"]
  }}
}}

Here the Extracted Text 

{ocr}

'''    
)

chain = LLMChain(llm=llm, prompt=prompt_template)

def extract_invoice_json_utility(ocr_text):
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
