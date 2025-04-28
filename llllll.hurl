# Test initializing a game with a location
GET http://localhost:8000/init/Copenhagen,%20Denmark
HTTP 200
[Asserts]
jsonpath "$.game_id" exists
[Captures]
game_id: jsonpath "$.game_id"

# Test resetting a game
GET http://localhost:8000/reset/{{game_id}}
HTTP 200
[Asserts]
jsonpath "$.state" exists

# Test stepping through the game
GET http://localhost:8000/step/{{game_id}}
HTTP 200
[Asserts]
jsonpath "$.state" exists

# Test stepping through the game again
GET http://localhost:8000/step/{{game_id}}
HTTP 200
[Asserts]
jsonpath "$.state" exists

# Test closing the game
POST http://localhost:8000/close/{{game_id}}
HTTP 200

# Test with a different location
GET http://localhost:8000/init/New%20York,%20USA
HTTP 200
[Asserts]
jsonpath "$.game_id" exists
[Captures]
game_id2: jsonpath "$.game_id"

# Test resetting the second game
GET http://localhost:8000/reset/{{game_id2}}
HTTP 200
[Asserts]
jsonpath "$.state" exists

# Test closing the second game
POST http://localhost:8000/close/{{game_id2}}
HTTP 200
