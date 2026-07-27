services:
  - type: web
    name: fachavi-sql-bot
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: VERIFY_TOKEN
        sync: false
      - key: WHATSAPP_TOKEN
        sync: false
      - key: PHONE_NUMBER_ID
        sync: false
      - key: ANTHROPIC_API_KEY
        sync: false
      - key: GOOGLE_CREDENTIALS_JSON
        sync: false
      - key: MASTER_SPREADSHEET_ID
        sync: false
      - key: GRAPH_API_VERSION
        value: v21.0
      - key: CLAUDE_MODEL
        value: claude-haiku-4-5-20251001
      - key: DATA_CACHE_TTL
        value: "60"
