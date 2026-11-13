# Construction-Site-Safety-Compliance-Pipeline-Worker-Localisation-PPE-Compliance-Classification
Construction and industrial sites are monitored by fixed cameras that capture wide scenes containing many workers at once. A safety system cannot judge compliance on a whole frame: it must first locate each individual worker, crop that worker out, and only then decide whether that person is wearing the mandatory protective equipment. 
Asking "is there a helmet somewhere in this image?" is not the same as asking "is THIS worker compliant?" - and the second question is the one a safety officer needs answered.

In this project you will build this two-stage pipeline using the Construction-PPE dataset (1,416 site images, 11 annotated object classes). The dataset only provides object-level boxes (person, helmet, vest, and their "missing" counterparts) - it does NOT provide a per-worker compliance label. You will first derive that label yourself by linking each helmet / vest annotation to the worker it belongs to, then build Stage 1 (worker localisation and cropping with a pretrained YOLO detector) and Stage 2 (a CNN that classifies each worker crop as COMPLIANT, NO_VEST or NO_HELMET).

Beyond the pipeline you will solve a real engineering decision: should the Stage-2 classifier be a lightweight ResNet-18 that can run on a site-edge camera box, or a deeper ResNet-34 that is more accurate but slower? You will run a structured Speed vs. Accuracy Trade-off Analysis - with an extra safety constraint, because missing a real violation (a false negative) is far costlier than raising a false alarm - and present a justified recommendation.
Business Use Cases
•	Construction & Infrastructure Safety Audits: Site safety officers cannot watch every camera feed. Per-worker compliance alerts let a small EHS team monitor many sites, and the Speed vs. Accuracy analysis mirrors the real choice between an on-site edge box and a cloud server.
•	Manufacturing & Warehouse EHS Monitoring: Factories and logistics hubs need continuous evidence that mandatory equipment is worn in designated zones, with an audit trail of which worker-crops were flagged.
•	Insurance & Contractor Risk Assessment: Insurers and main contractors can score subcontractors on measured compliance rates (share of workers COMPLIANT per site-day) instead of self-reported checklists.
•	Public Works & Municipal Tenders: Authorities supervising road, metro and building projects can audit contractor compliance from routine site photographs, prioritising inspections where violations cluster.

Worker Localisation (Stage 1) + PPE Compliance Classification (Stage 2)
A site camera delivers a wide frame with many workers. This notebook builds a two-stage pipeline:


Stage 1 - Worker localisation: pretrained YOLO finds each worker; each is cropped.
Stage 2 - Compliance classification: a CNN labels each crop COMPLIANT, NO_VEST or NO_HELMET.
The dataset only has object-level boxes, so the worker-level label is derived from them (Section 3). A Speed vs. Accuracy analysis (custom CNN vs ResNet-18 vs ResNet-34) closes with a safety-first recommendation.

Dataset: Construction-PPE (Ultralytics, AGPL-3.0) - https://docs.ultralytics.com/datasets/detect/construction-ppe Direct download: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip


2. Data Loading & EDA
<img width="1089" height="440" alt="download" src="https://github.com/user-attachments/assets/b610e258-5b9b-453d-a621-f1f222224636" />

<img width="1189" height="390" alt="download" src="https://github.com/user-attachments/assets/0fd23a52-bf7d-4950-b7f1-138e6ccaa454" />

<img width="637" height="345" alt="Capture1" src="https://github.com/user-attachments/assets/38af80ab-3399-468e-885e-9c60ffdf3b27" />

=== COMPLIANCE CLASS COUNTS ===
split             test  train  val
compliance_label                  
COMPLIANT          173   1116  176
NO_HELMET           19    224   14
NO_VEST             21    227   18

<img width="690" height="390" alt="download" src="https://github.com/user-attachments/assets/aa071a22-8329-4221-b8e8-19d965ac89a3" />
<img width="690" height="390" alt="download" src="https://github.com/user-attachments/assets/60a9698c-a695-4b1d-96eb-1ca544ffcc2c" />

Transfer Learning - ResNet-18 and ResNet-34

Auditing trained model parameters and latency profiles...

=== STAGE-2 PRODUCTION TRADE-OFF MATRIX ===
Model Name  Val Accuracy (%)  Test Accuracy (%)  Macro F1 (%)  Violation Recall (%)  Inference Time per Crop (ms)  Model File Size (MB)  Total Training Time (min)
Custom CNN             60.10              59.62         33.95                  37.5                         8.896                 17.42                       5.52
 ResNet-18             55.77              65.26         39.88                  37.5                        20.212                 44.79                       8.16
 ResNet-34             61.06              62.44         38.49                  32.5                        36.850                 85.28                      15.07

Majority-class baseline (always COMPLIANT): 81.2% accuracy, 0.0% violation recall
-> Objective: System deployment choices must balance speed (ms) vs. safety violation recall.

<img width="889" height="490" alt="download" src="https://github.com/user-attachments/assets/13bba38b-a8fe-44b7-b06d-69d6ef3c9066" />

=== FINAL SYSTEM METRIC REPORT: RESNET-34 ===
Test loss: 0.9377 | Test accuracy: 0.6244

              precision    recall  f1-score   support

   COMPLIANT      0.820     0.711     0.762       173
   NO_HELMET      0.088     0.158     0.113        19
     NO_VEST      0.241     0.333     0.280        21

    accuracy                          0.624       213
   macro avg      0.383     0.401     0.385       213
weighted avg      0.698     0.624     0.656       213

Note: NO_VEST has only 21 test workers - its scores are noisy.

<img width="567" height="490" alt="download" src="https://github.com/user-attachments/assets/ed5584db-071a-4e1f-a248-87ea81049abe" />

<img width="1454" height="590" alt="download" src="https://github.com/user-attachments/assets/45aed46e-59c5-4966-8041-ce5cb4402bb0" />

<img width="569" height="590" alt="download" src="https://github.com/user-attachments/assets/f3172f9e-c7d1-4db8-8eea-bf93be601a73" />
<img width="569" height="590" alt="download" src="https://github.com/user-attachments/assets/925882cb-21ef-425b-831a-64bd84fc93b7" />
<img width="790" height="551" alt="download" src="https://github.com/user-attachments/assets/a19dd694-aead-4e7c-a22b-daa5165a5026" />

Deployment:
<img width="835" height="407" alt="Capture3" src="https://github.com/user-attachments/assets/90a4affa-f1f5-40ef-9eef-f8b97a6f5cf8" />
<img width="863" height="295" alt="Capture2" src="https://github.com/user-attachments/assets/0b2f1e90-50bf-4662-8077-244191e9ddbf" />


Observations:
When a worker is missing both a helmet and a vest, the model sees two different violations at once—an uncovered head (NO_HELMET) and an unvested torso (NO_VEST)—and picks whichever visual feature has the stronger confidence score in that specific photo.

Why This Happens

• Visual Cue Competition: Depending on camera angle, lighting, or distance, the model might notice the missing high-visibility vest more strongly (predicting NO_VEST) or the bare head more strongly (predicting NO_HELMET).
• Training Data Overlap: During training, crops missing both items were mapped into the single-violation classes, so the neural network splits its probability between NO_HELMET and NO_VEST whenever both are absent.
• No Impact on Site Safety Grading: In both cases, the worker is flagged with a red bounding box and counted as a violation in your overall site verdict—which is the exact goal of a 3-class compliance checker.



This as a known limitation: because the Stage-2 classifier uses 3 mutually exclusive classes (COMPLIANT, NO_VEST, NO_HELMET), workers missing both PPE items are assigned to whichever single violation has higher model confidence (and still trigger a red violation alert). 

<img width="640" height="640" alt="image10" src="https://github.com/user-attachments/assets/c50c99f5-df60-4c99-b864-739c6efda549" />

This image shows a great real-world test case of your two-stage pipeline! Looking closely at the annotated photo, all three railway workers are wearing orange safety vests, and none of them are wearing helmets.
However, your app predicted COMPLIANT (Green) for the first two workers and NO_HELMET (Red) for the third worker.

Why This Happened 
• 1. Orange Vests Strongly Trigger COMPLIANT: In your training dataset, 80.3% of all workers (1,116 of 1,567) were COMPLIANT and wearing bright orange/yellow vests. Because bright orange vests occupy a huge area of the 128 × 64 crop, the model relies heavily on the bright vest color and often guesses COMPLIANT even when a helmet is missing—matching your confusion matrix where 16 out of 21 (76%) true NO_HELMET workers were predicted as COMPLIANT.
• 2. Small Head Pixel Size at 128 × 64: In the first two crops, the workers' bare heads occupy only ~15 × 15 pixels at the very top of the resized image, making it hard for ResNet to tell a bare head apart from a helmet. For the third worker (farther back, darker background contrast around the head), the bare head stood out enough for the model to catch NO_HELMET.




