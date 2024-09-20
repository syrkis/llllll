# Test assigning a cookie
POST http://localhost:8000/assign_cookie
HTTP 200
[Asserts]
header "Set-Cookie" exists
jsonpath "$.session_id" exists
[Captures]
session_id: header "Set-Cookie" regex "session_id=([^;]+)"

# Test starting a game
POST http://localhost:8000/start_game
Cookie: session_id={{session_id}}
{ "scenario" : "2s3z" }
HTTP 200

#
