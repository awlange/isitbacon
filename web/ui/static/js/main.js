$(document).ready(function() {

    var fileSizeLimit = 1024 * 1024 * 2;  // 2 MB limit

    //$("form").submit(function(event) {
    //    event.preventDefault();
    //});

    // Displaying image on upload
    $("input").change(function(e) {
        for (var i = 0; i < e.originalEvent.srcElement.files.length; i++) {

            var file = e.originalEvent.srcElement.files[i];
            var submit = $("#submit-button");
            var img = $("#image-display");
            var errorRow = $("#error-row");

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
});