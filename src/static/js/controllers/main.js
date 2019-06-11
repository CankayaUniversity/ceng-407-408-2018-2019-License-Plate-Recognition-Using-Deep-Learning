meanStackTemplate.controller("mainController", function ($http, $scope) {
    $scope.PreviewImage = null;
    $scope.Image = null
    $scope.httpRequestWaiting = true;
    $scope.httpRequestFailed = false;

    $scope.letters=null;
    $scope.charimages = null;
    $scope.filepath = null;

    $scope.SelectFile = function (e) {
        $scope.Image = new FormData();
        $scope.Image.append("file", e.target.files[0]);
        var reader = new FileReader();
        reader.onload = function (e) {
            $scope.PreviewImage = e.target.result;
            $scope.$apply();
        };
        reader.readAsDataURL(e.target.files[0]);
    }
    $scope.submitFile = function () {

        var request = {
            method: 'POST',
            url: '/upload',
            data: $scope.Image,
            headers: {
                'Content-Type': undefined
            }
        };

        $http(request)
            .then(function (response) {
                $scope.httpRequestWaiting = false;
                $scope.httpRequestFailed = false;
                $scope.filepath = response.data.filepath;
                $scope.charimages = response.data.char_images;
                $scope.letters = response.data.letters;
                console.log(response.data)
            },function (error) {
                $scope.httpRequestWaiting = false;
                $scope.httpRequestFailed = true;
                
            });
    }
    $scope.closeFile = function (e) {
        $scope.PreviewImage = null;
        $scope.Image = null
        $scope.httpRequestWaiting = true;
        $scope.httpRequestFailed = false;
    
        $scope.charimages = null;
        $scope.filepath = null;
    }

});