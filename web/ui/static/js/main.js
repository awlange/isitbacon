var _URL = window.URL || window.webkitURL;

$(document).ready(function() {

    var fileSizeLimit = 1024 * 1024 * 2;  // 2 MB limit
    var file = null;

    // Displaying image on upload
    $("#upload-button").change(function(e) {
        for (var i = 0; i < e.originalEvent.srcElement.files.length; i++) {

            file = e.originalEvent.srcElement.files[i];
            var submit = $("#submit-button");
            var img = $("#image-display");
            var errorRow = $("#error-row");

            // Hide data if showing
            var dataRow = $("#data-row");
            if (!dataRow.hasClass("hide")) {
                dataRow.addClass("hide");
            }

            if (file.size < fileSizeLimit) {
                // Only allow if file is less than 2 MB

                if (!errorRow.hasClass("hide")) {
                    errorRow.addClass("hide");
                }

                var reader = new FileReader();
                reader.onloadend = function() {
                     img.attr("src", reader.result);
                };
                reader.readAsDataURL(file);

                // Enable submit button
                submit.removeClass("btn-default");
                submit.addClass("btn-success");
                submit.removeClass("disabled");
                submit.addClass("enabled");

            } else {
                // File is too big. Write an error message.
                file = null;

                // Display the error message
                errorRow.removeClass("hide");

                // Clear img
                img.attr("src", "");

                // Disable submit button if enabled
                if (submit.hasClass("enabled")) {
                    submit.removeClass("btn-success");
                    submit.addClass("btn-default");
                    submit.removeClass("enabled");
                    submit.addClass("disabled");
                }
            }
        }
    });

    // Image submission
    $("#submit-button").click(function() {
        if ($(this).hasClass("disabled") || file == null) {
            return;
        }

        // Add spinner
        var spinner = $("#spinner-row");
        if (spinner.hasClass("hide")) {
            spinner.removeClass("hide");
        }

        // AJAX post of the image
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "/submit", true);
        xhr.setRequestHeader("Content-Type", "image/jpeg");
        xhr.setRequestHeader("X_FILENAME", file.name);
        xhr.setRequestHeader("X_foobar", "supersecret");
        xhr.onload = function() {
            if (this.status == 200) {
                console.log(this.responseText);
                data = JSON.parse(this.responseText);
                setTimeout(function(){ displayData(data); }, 800);  // wait a little for dramatic effect ;)
            }
        };
        xhr.send(file);
    });
});

function displayData(data) {
    // Display BaconNet results

    var spinner = $("#spinner-row");
    if (!spinner.hasClass("hide")) {
        spinner.addClass("hide");
    }

    var dataRow = $("#data-row");
    if (dataRow.hasClass("hide")) {
        dataRow.removeClass("hide");
    }
}