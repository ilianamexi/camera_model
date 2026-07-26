import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# --- Βήμα 1: Ανάγνωση αρχείου STL ---
filename = 'man_small.stl'

# Χρησιμοποιούμε τη βιβλιοθήκη numpy-stl
try:
    my_mesh = mesh.Mesh.from_file(filename)
    print("Το αρχείο STL φορτώθηκε επιτυχώς.")
except FileNotFoundError:
    print(f"Σφάλμα: Το αρχείο {filename} δεν βρέθηκε.")
    exit()

# Τα vertices στο numpy-stl είναι οργανωμένα ανά τρίγωνο (Faces x 3 x 3)
# Για την εργασία, χρειαζόμαστε μια λίστα σημείων.
# Παίρνουμε όλα τα σημεία (μπορεί να υπάρχουν διπλότυπα κορυφών, αλλά για την προβολή δεν πειράζει)
vertices = my_mesh.vectors.reshape(-1, 3)
print(f"Αριθμός σημείων (Vertices): {vertices.shape[0]}")

# --- Βήμα 2: Οπτικοποίηση αρχικού αντικειμένου ---
# Σημείωση: Η 3D απεικόνιση πολλών τριγώνων στο Matplotlib είναι αργή.
# Εδώ κάνουμε μια απλή απεικόνιση σημείων (scatter) για ταχύτητα, 
# ή χρησιμοποιούμε slice για να μην κολλήσει ο υπολογιστής.
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')

# Δειγματοληψία σημείων (κάθε 10ο σημείο) για γρήγορη απεικόνιση
ax1.scatter(vertices[::10, 0], vertices[::10, 1], vertices[::10, 2], s=1, c='g')

ax1.set_title('Σχήμα 1: Αρχικό Αντικείμενο')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# --- ΟΡΙΣΜΟΣ ΟΡΙΩΝ ΑΞΟΝΩΝ (-0.8 έως 0.8) ---
ax1.set_xlim(-0.8, 0.8)
ax1.set_ylim(-0.8, 0.8) 
ax1.set_zlim(-0.8, 0.8)

# Ισομετρική προβολή (για να μην παραμορφώνεται)
ax1.set_box_aspect([1,1,1]) 





# --- Βήμα 3: Δημιουργία 4 ανθρώπων (Σκηνή) ---
num_copies = 4
offset_step = 1.0  # Απόσταση μεταξύ τους
all_points_list = []

for i in range(num_copies):
    current_verts = vertices.copy()
    
    # ΑΛΛΑΓΗ: Χρησιμοποιούμε += για να πάνε στο 0, 1, 2, 3 (όπως φαίνεται στο PDF)
    current_verts[:, 0] += (i * offset_step) 
    
    all_points_list.append(current_verts)

# Συγχώνευση
all_vertices = np.vstack(all_points_list)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')

# Απεικόνιση (με δειγματοληψία για ταχύτητα)
ax2.scatter(all_vertices[::20, 0], all_vertices[::20, 1], all_vertices[::20, 2], s=0.5, c='k')

ax2.set_title('Σχήμα 2: 4 Άνθρωποι')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# -- ΟΡΙΣΜΟΣ ΟΡΙΩΝ ΓΙΑ ΝΑ ΤΑΙΡΙΑΖΕΙ ΜΕ ΤΗΝ ΕΙΚΟΝΑ
# Στην εικόνα του PDF ο άξονας X έχει ταμπελάκια 3, 2, 1, 0, -1.
ax2.set_xlim(-1, 3.5) 
ax2.set_ylim(-1, 1)    
ax2.set_zlim(-1, 1)

# Σωστή αναλογία διαστάσεων
ax2.set_box_aspect([1, 0.5, 0.5]) # Προσαρμογή για να μοιάζει "μακρόστενο" στον άξονα X

# Ρύθμιση γωνίας θέασης (View) για να μοιάζει με την εκφώνηση
ax2.view_init(elev=20, azim=-60)




# --- Βήμα 4: Πίνακας Εξωτερικών Παραμέτρων (Extrinsics) ---
# Ζητείται: Περιστροφή Ry(-90) και Θέση Κάμερας C = [-2, 0.8, 0]

# --- Βήμα 4: Πίνακας Εξωτερικών Παραμέτρων (ΔΙΟΡΘΩΜΕΝΟ) ---
# Ζητείται: R = Ry(-90) * Rx(90)
# Αυτό είναι απαραίτητο για να γυρίσει το αντικείμενο από το "Z-up" (κόσμος) στο "Y-up" (κάμερα)

# 1. Υπολογισμός Ry(-90)
theta_y_deg = -90
rad_y = np.radians(theta_y_deg)
cy, sy = np.cos(rad_y), np.sin(rad_y)
Ry = np.array([
    [cy,  0, sy],
    [0,   1, 0],
    [-sy, 0, cy]
])

# 2. Υπολογισμός Rx(90)
theta_x_deg = 90
rad_x = np.radians(theta_x_deg)
cx, sx = np.cos(rad_x), np.sin(rad_x)
Rx = np.array([
    [1,  0,   0],
    [0, cx, -sx],
    [0, sx,  cx]
])

# 3. Τελικός Πίνακας Περιστροφής: R = Ry * Rx
# Η σειρά πολλαπλασιασμού είναι σημαντική!
R = Ry @ Rx

# Θέση Κάμερας (Camera Center)
C = np.array([-2, 0.8, 0]).reshape(3, 1)

# Υπολογισμός Translation vector t = -R * C
t = -R @ C

print("\n--- Αποτελέσματα Υπολογισμών (Updated) ---")
print("Πίνακας Περιστροφής R:\n", R)
print("Διάνυσμα Μετάθεσης t:\n", t)

# Δημιουργία πίνακα P_extr (4x4)
P_extr = np.eye(4)
P_extr[:3, :3] = R
P_extr[:3, 3] = t.flatten()




# --- Βήμα 5: Προβολή και Οπτικοποίηση (ΠΛΗΡΕΣ & ΔΙΟΡΘΩΜΕΝΟ) ---

# 1. Δημιουργία του πίνακα Q (Ομογενείς Συντεταγμένες)
# Πρέπει πρώτα να φτιάξουμε το Q για να μπορέσουμε να το πολλαπλασιάσουμε!
N = all_vertices.shape[0]
ones_col = np.ones((N, 1))
Q = np.hstack([all_vertices, ones_col]) # Ενωση σημείων με τη στήλη '1'

# Κάνουμε ανάστροφο (Transpose) ώστε να γίνει 4xN
Q = Q.T 

# 2. Προβολή: Q' = P_extr * Q
# Τώρα το Q είναι ορισμένο, οπότε η πράξη θα γίνει κανονικά.
Q_prime = P_extr @ Q

# 3. Ανάκτηση συντεταγμένων στο σύστημα της κάμερας
x_prime = Q_prime[0, :]
y_prime = Q_prime[1, :]
z_prime = Q_prime[2, :]

# 4. Φιλτράρισμα σημείων (Clipping)
# Κρατάμε μόνο τα σημεία που είναι ΜΠΡΟΣΤΑ από την κάμερα (z' > 0.1)
# Αυτό λύνει το πρόβλημα με τις τεράστιες τιμές στους άξονες.
valid_points_mask = z_prime > 0.1

if np.sum(valid_points_mask) == 0:
    print("Προσοχή: Κανένα σημείο δεν βρίσκεται μπροστά από την κάμερα!")
else:
    # Εφαρμογή της διαίρεσης ΜΟΝΟ στα έγκυρα σημεία
    u = x_prime[valid_points_mask] / z_prime[valid_points_mask]
    v = y_prime[valid_points_mask] / z_prime[valid_points_mask]

    # 5. Γραφική παράσταση (Σχήμα 3)
    plt.figure(3)
    plt.plot(u, v, 'k.', markersize=0.5)
    plt.title('Σχήμα 3: Προβολή Κάμερας (Τελικό)')
    plt.xlabel('u (image)')
    plt.ylabel('v (image)')
    plt.axis('equal')
    plt.grid(True)
    # Αντιστροφή άξονα Y (συχνά στις εικόνες το (0,0) είναι πάνω αριστερά)
    plt.gca().invert_yaxis()

    plt.show()