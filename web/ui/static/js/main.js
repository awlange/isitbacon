$(document).ready(function() {

    //$("form").submit(function(event) {
    //    event.preventDefault();
    //});

    // Displaying image on upload
    $("input").change(function(e) {
        for (var i = 0; i < e.originalEvent.srcElement.files.length; i++) {
            var file = e.originalEvent.srcElement.files[i];
            var img = $("#image-display");
            var reader = new FileReader();
            reader.onloadend = function() {
                 img.attr("src", reader.result);
            };
            reader.readAsDataURL(file);

            // Enable submit button
            var submit = $("#submit-button");
            submit.removeClass("btn-default");
            submit.addClass("btn-success");
            submit.removeClass("disabled");
            submit.addClass("enabled");
        }
    });
});