# Data Review Guidelines for Freelancers

This document explains how to review and clean the dataset collected by the VisionTera AI system.

**Your Role:** Quality Control. The AI automatically collects and attempts to label images. Your job is to fix its mistakes.

## 1. Where to Work

All work happens in this folder:
`datasets/to_label/`

Inside, you will find three folders:
1.  📂 **`MALE`** (AI thinks these are Male)
2.  📂 **`FEMALE`** (AI thinks these are Female)
3.  📂 **`UNCERTAIN`** (AI is not sure)

---

## 2. Your Tasks (Step-by-Step)

### Step 1: Review the MALE Folder
1.  Open `datasets/to_label/MALE`.
2.  Look through **all** images.
3.  **Action**:
    *   If you see a **Female**, move the image to the `FEMALE` folder.
    *   If you see a **non-person** (chair, shadow, empty wall), **DELETE** it.
    *   If the image is extremely blurry or you can't tell, **DELETE** it.

### Step 2: Review the FEMALE Folder
1.  Open `datasets/to_label/FEMALE`.
2.  Look through **all** images.
3.  **Action**:
    *   If you see a **Male**, move the image to the `MALE` folder.
    *   If you see a **non-person**, **DELETE** it.
    *   If the image is extremely blurry or you can't tell, **DELETE** it.

### Step 3: Sort the UNCERTAIN Folder
1.  Open `datasets/to_label/UNCERTAIN`.
2.  These are images the AI found difficult.
3.  **Action**:
    *   Identify the person's gender.
    *   **Move** Males to the `MALE` folder.
    *   **Move** Females to the `FEMALE` folder.
    *   **DELETE** anything else (blurry, unknown, not a person).

---

## 3. Important Rules

1.  **Quality over Quantity**: It is better to DELETE a bad image than to keep a confusing one.
    *   *Rule of Thumb: If you cannot tell the gender within 2 seconds, DELETE it.*
2.  **One Person Only**: The image should ideally show one clear person. If there are multiple people and it's confusing who is the subject, DELETE it.
3.  **Continuous Work**: The system is running in the background and adding new images 24/7.
    *   You might see new files appear while you work. This is normal.
    *   Just focus on clearing the folders one by one.

## 4. Summary of Classes

| Class | Description |
| :--- | :--- |
| **MALE** | Subject is clearly male (clothing, appearance). |
| **FEMALE** | Subject is clearly female. (Includes Abaya/Hijab). |
| **DELETE** | Blurry, too dark, not a person, or impossible to tell. |
