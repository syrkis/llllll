// src/lib/piece.ts
import { writable } from "svelte/store";
import type { ChessPiece } from "./types";

// Configure an initial array of chess pieces
const initialPieces: ChessPiece[] = [
  { name: "King", symbol: "♔", x: 0, y: 0, active: false },
  { name: "Queen", symbol: "♕", x: 0, y: 0, active: false },
  { name: "Rook", symbol: "♖", x: 0, y: 0, active: false },
  { name: "Bishop", symbol: "♗", x: 0, y: 0, active: false },
  { name: "Knight", symbol: "♘", x: 0, y: 0, active: false },
  { name: "Pawn", symbol: "♙", x: 0, y: 0, active: false },
];

// Create a writable store to manage chess pieces
export const piecesStore = writable<ChessPiece[]>(initialPieces);

// Function to handle click events on pieces
export function handlePieceClick(x: number, y: number, threshold: number) {
  piecesStore.update((pieces) => {
    let pieceIndex = pieces.findIndex(
      (p) => p.active && Math.abs(p.x - x) <= threshold && Math.abs(p.y - y) <= threshold,
    );
    if (pieceIndex !== -1) {
      pieces[pieceIndex].active = false;
    } else {
      pieceIndex = pieces.findIndex((p) => !p.active);
      if (pieceIndex !== -1) {
        pieces[pieceIndex].x = x;
        pieces[pieceIndex].y = y;
        pieces[pieceIndex].active = true;
      }
    }
    return pieces;
  });
}

// Function to handle piece dragging operations
export function handlePieceDrag(isDragging: { value: boolean }, dragIndex: { value: number }, x: number, y: number) {
  if (isDragging.value) {
    piecesStore.update((pieces) => {
      if (dragIndex.value !== -1) {
        pieces[dragIndex.value].x += x;
        pieces[dragIndex.value].y += y;
      }
      return pieces;
    });
  }
}
