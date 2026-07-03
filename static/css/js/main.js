/* ==========================================================
   AGRONETRA PREMIUM UI
   MAIN.JS
========================================================== */

document.addEventListener("DOMContentLoaded", () => {

    /* ==========================================
        ELEMENTS
    ========================================== */

    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("overlay");
    const menuBtn = document.getElementById("menuBtn");
    const scrollBtn = document.getElementById("scrollTopBtn");
    const loader = document.getElementById("loading-screen");

    /* ==========================================
        SIDEBAR
    ========================================== */

    function openSidebar() {

        if (!sidebar) return;

        sidebar.classList.add("active");

        overlay.classList.add("active");

        document.body.style.overflow = "hidden";

    }

    function closeSidebar() {

        if (!sidebar) return;

        sidebar.classList.remove("active");

        overlay.classList.remove("active");

        document.body.style.overflow = "auto";

    }

    if (menuBtn) {

        menuBtn.addEventListener("click", openSidebar);

    }

    if (overlay) {

        overlay.addEventListener("click", closeSidebar);

    }

    document.addEventListener("keydown", (e) => {

        if (e.key === "Escape") {

            closeSidebar();

        }

    });

    /* ==========================================
        SCROLL TOP
    ========================================== */

    window.addEventListener("scroll", () => {

        if (!scrollBtn) return;

        if (window.scrollY > 250) {

            scrollBtn.style.display = "flex";

        } else {

            scrollBtn.style.display = "none";

        }

    });

    if (scrollBtn) {

        scrollBtn.addEventListener("click", () => {

            window.scrollTo({

                top: 0,

                behavior: "smooth"

            });

        });

    }

    /* ==========================================
        PAGE LOADER
    ========================================== */

    function showLoader() {

        if (loader) {

            loader.style.display = "flex";

        }

    }

    function hideLoader() {

        if (loader) {

            loader.style.display = "none";

        }

    }

    window.addEventListener("load", () => {

        hideLoader();

    });

    document.querySelectorAll("form").forEach(form => {

        form.addEventListener("submit", () => {

            showLoader();

        });

    });

    /* ==========================================
        BUTTON RIPPLE EFFECT
    ========================================== */

    const buttons = document.querySelectorAll(".btn");

    buttons.forEach(button => {

        button.addEventListener("click", function (e) {

            const ripple = document.createElement("span");

            const rect = this.getBoundingClientRect();

            const size = Math.max(rect.width, rect.height);

            ripple.style.width = ripple.style.height = size + "px";

            ripple.style.left = (e.clientX - rect.left - size / 2) + "px";

            ripple.style.top = (e.clientY - rect.top - size / 2) + "px";

            ripple.classList.add("ripple");

            this.appendChild(ripple);

            setTimeout(() => {

                ripple.remove();

            }, 600);

        });

    });

    /* ==========================================
        ACTIVE NAVIGATION
    ========================================== */

    const current = window.location.pathname;

    document.querySelectorAll(".nav-item").forEach(item => {

        const href = item.getAttribute("href");

        if (!href) return;

        if (current === href) {

            item.style.color = "#2E7D32";

            item.style.fontWeight = "600";

        }

    });

    document.querySelectorAll(".menu a").forEach(item => {

        const href = item.getAttribute("href");

        if (!href) return;

        if (current === href) {

            item.style.background = "#2E7D32";

            item.style.color = "#fff";

            const icon = item.querySelector("i");

            if (icon) {

                icon.style.color = "#fff";

            }

        }

    });

    /* ==========================================
        CARD ANIMATION
    ========================================== */

    const observer = new IntersectionObserver(entries => {

        entries.forEach(entry => {

            if (entry.isIntersecting) {

                entry.target.classList.add("fade-up");

            }

        });

    }, {

        threshold: 0.15

    });

    document.querySelectorAll(".card,.stat-card,.history-card,.analytics-card,.profile-card,.weather-card,.result-card").forEach(card => {

        observer.observe(card);

    });

    /* ==========================================
        IMAGE PREVIEW
    ========================================== */

    const imageInput = document.querySelector("input[type=file]");

    const preview = document.getElementById("previewImage");

    if (imageInput && preview) {

        imageInput.addEventListener("change", function () {

            const file = this.files[0];

            if (!file) return;

            const reader = new FileReader();

            reader.onload = function (e) {

                preview.src = e.target.result;

                preview.style.display = "block";

            }

            reader.readAsDataURL(file);

        });

    }

    /* ==========================================
        NOTIFICATION BUTTON
    ========================================== */

    const notification = document.querySelector(".notification-btn");

    if (notification) {

        notification.addEventListener("click", () => {

            alert("No new notifications.");

        });

    }

    /* ==========================================
        FAB CLICK ANIMATION
    ========================================== */

    const fab = document.querySelector(".fab");

    if (fab) {

        fab.addEventListener("click", () => {

            fab.style.transform = "translateX(-50%) scale(.90)";

            setTimeout(() => {

                fab.style.transform = "translateX(-50%) scale(1)";

            }, 150);

        });

    }

});

/* ==========================================================
    RIPPLE CSS (Injected Automatically)
========================================================== */

const style = document.createElement("style");

style.innerHTML = `

.btn{

position:relative;

overflow:hidden;

}

.ripple{

position:absolute;

border-radius:50%;

background:rgba(255,255,255,.5);

transform:scale(0);

animation:ripple .6s linear;

pointer-events:none;

}

@keyframes ripple{

to{

transform:scale(4);

opacity:0;

}

}

`;

document.head.appendChild(style);

/* ==========================================================
    END OF MAIN.JS
========================================================== */